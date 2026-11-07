"""Continue a verified Muon checkpoint through three complete frozen RGB passes."""

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.data import DeterministicBatches, check_split, read_manifest
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.optimizer_trial import optimizer_counts, policy
from radio_moge.rgb_input_audit import RGBInputAudit, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords, validate_replacement
import radio_moge.train as trainer
from prepare_room_rgb_pool import check_rows
from run_room_feature_epoch import IMAGES, SOURCES, real_cursor
from run_room_optimizer_arm import OptimizerArmAudit, cpu
from run_room_true_batch_epoch import BatchCoverage, BUCKET_POLICY
from verify_corpus_stage_startup import equal_finite

BATCH = 128
EPOCHS = 3
UPDATES_PER_EPOCH = (IMAGES + BATCH // 8 - 1) // (BATCH // 8)
UPDATES = EPOCHS * UPDATES_PER_EPOCH
SECONDS = 36000
CADENCE = 500
COVERAGE_GOAL = "three_complete_frozen_synthetic_passes_with_per_pass_tails"


def endpoints(source_step, batch=BATCH):
    if batch != BATCH:
        raise ValueError("Actual128 is required")
    return [source_step + n for n in range(CADENCE, UPDATES, CADENCE)] + [source_step + UPDATES]


def epoch_shape(plan):
    expected = dict(
        actual_batch=BATCH,
        additional_updates=UPDATES,
        epochs=EPOCHS,
        updates_per_epoch=UPDATES_PER_EPOCH,
        synthetic_images=IMAGES,
        synthetic_presentations=IMAGES * EPOCHS,
        coverage_goal=COVERAGE_GOAL,
    )
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Wrong declared continuation budget or population")
    return BATCH, UPDATES, CADENCE


class MultiEpochBatches:
    """Continue original keys; independently permute each complete synthetic pass.

    The last update of EACH pass has three synthetic and125 original images.
    New passes get different permutation seeds; original crop seeds continue.
    Reading unused prefetch past the final pass never repeats synthetic images.
    """

    def __init__(
        self,
        records,
        spec,
        batch_size,
        buckets,
        seed,
        start,
        count,
        actual_batch=BATCH,
        epochs=EPOCHS,
    ):
        if (
            not isinstance(records, ReplacementRecords)
            or batch_size != 4
            or actual_batch != BATCH
            or spec["every_images"] != 8
            or epochs != EPOCHS
        ):
            raise ValueError("Unexpected multi-pass sampler configuration")
        self.records, self.spec, self.buckets = records, spec, buckets
        self.seed, self.start, self.count = seed, start, count
        self.groups_per_update = actual_batch // batch_size
        self.groups_per_epoch = ((records.synthetic_size + 15) // 16) * self.groups_per_update

    def __len__(self):
        return self.count

    def __iter__(self):
        real = DeterministicBatches(
            self.records.real_size,
            4,
            self.buckets,
            self.seed,
            self.spec["real_start_cursor"] + self.start,
            self.count,
        )
        previous, order = None, None
        for absolute, batch in enumerate(real, self.start):
            epoch, group = divmod(absolute, self.groups_per_epoch)
            if epoch != previous:
                order = list(range(self.records.synthetic_size))
                random.Random(self.spec["seed"] + epoch).shuffle(order)
                previous = epoch
            first = (
                self.spec["real_start_cursor"]
                + (absolute // self.groups_per_update) * self.groups_per_update
            )
            height, width = random.Random(self.seed + first * 104729).choice(self.buckets)
            result = []
            for offset, key in enumerate(batch):
                slot = group * 4 + offset
                if epoch < EPOCHS and (slot + 1) % 8 == 0 and slot // 8 < len(order):
                    key = (self.records.real_size + order[slot // 8], *key[1:])
                result.append((key[0], key[1], height, width))
            yield result


class MultiEpochCoverage:
    def __init__(self, real_size, synthetic, batch=BATCH):
        self.trackers = [BatchCoverage(real_size, synthetic, batch) for _ in range(EPOCHS)]
        self.width = (len(synthetic) + batch // 8 - 1) // (batch // 8)
        self.updates = 0

    def validate(self, rows):
        epoch = self.updates // self.width
        if epoch >= EPOCHS:
            raise ValueError("Training exceeded the authorized three passes")
        return self.trackers[epoch].validate(rows)

    def commit(self, indices):
        epoch = self.updates // self.width
        self.trackers[epoch].commit(indices)
        self.updates += 1
        if (
            self.updates % self.width == 0
            and not self.trackers[epoch].report()["each_image_exactly_once"]
        ):
            raise ValueError("A finished pass omitted a synthetic image")

    @property
    def counts(self):
        return [t.counts for t in self.trackers]

    def report(self):
        rows = [t.report() for t in self.trackers]
        return dict(
            epochs=rows,
            completed_epochs=sum(r["each_image_exactly_once"] for r in rows),
            committed=sum(r["committed"] for r in rows),
            expected=EPOCHS * len(self.trackers[0].counts),
            repeated_within_epochs=sum(r["repeated"] for r in rows),
            every_image_once_per_epoch=all(r["each_image_exactly_once"] for r in rows),
        )


def check_continued_counts(counts, source, updates):
    if (
        source["parameters"] != 100
        or source["core_parameters"] != 16
        or source["auxiliary_parameters"] != 84
        or source["adam_states"] != 84
        or source["muon_states"] != 16
        or source["core_adam_steps"]
        or len(source["auxiliary_adam_steps"]) != 1
        or source["muon_group_steps"] != [source["new_optimizer_steps"]]
        or source["new_optimizer_steps"] < 1
    ):
        raise ValueError("Source is not a trained16-Muon/84-Adam checkpoint")
    expected = deepcopy(source)
    expected["auxiliary_adam_steps"] = [source["auxiliary_adam_steps"][0] + updates]
    expected["muon_group_steps"] = [source["new_optimizer_steps"] + updates]
    expected["new_optimizer_steps"] += updates
    if counts != expected:
        raise ValueError(f"Muon or auxiliary Adam state was reset or miscounted: {counts}")


def initial_state_check(value, parent, config, performance):
    if (
        value["step"] != parent["step"]
        or value["cursor"] != 0
        or value["config"] != config
        or value["performance"] != performance
    ):
        raise ValueError("Actual initial cursor, configuration or batch profile differs")
    sections = (
        "adapter",
        "optimizer",
        "statistics",
        "rng",
        "adapter_config",
        "model_config",
        "revisions",
    )
    for name in sections:
        if not equal_finite(cpu(value[name]), parent[name]):
            raise ValueError("Continued initial state differs: " + name)
    check_continued_counts(
        optimizer_counts(value["optimizer"]), optimizer_counts(parent["optimizer"]), 0
    )
    scheduler = value["scheduler"]
    if (
        scheduler["base_lrs"] != [1e-5]
        or scheduler["_last_lr"] != [1e-5]
        or scheduler["last_epoch"] != 0
        or scheduler["_step_count"] != 1
    ):
        raise ValueError("Only the explicit constant-scheduler counter reset is permitted")
    return sections


def check_contract(config, parent, plan):
    epoch_shape(plan)
    validate_replacement(config)
    expected = deepcopy(parent["config"])
    expected["output_dir"] = config["output_dir"]
    expected["tracking"] = config["tracking"]
    expected["train"].update(
        steps=parent["step"] + UPDATES,
        max_seconds=SECONDS,
        eval_every=CADENCE,
        save_every=CADENCE,
        eval_at_start=False,
    )
    expected["data"]["rgb_replacement"]["real_start_cursor"] = real_cursor(parent)
    old_metrics = expected["evaluation"]["checkpoint_metrics"]
    new_metrics = config["evaluation"]["checkpoint_metrics"]
    if set(old_metrics) != set(new_metrics):
        raise ValueError("Evaluation checkpoint metrics changed")
    for name in old_metrics:
        old_metrics[name]["baseline"] = new_metrics[name]["baseline"]
    if config != expected:
        raise ValueError("Continuation changed more than budget, stream cursor and tracking")
    if (
        plan.get("authorized") is not True
        or plan["source_step"] != parent["step"]
        or plan["source_parent_cursor"] != parent["cursor"]
        or plan["source_cursor"] != real_cursor(parent)
        or plan["maximum_training_seconds"] != SECONDS
        or plan["bucket_policy"] != BUCKET_POLICY
        or config["optimizer_trial"] != policy("core-muon")
        or config["train"]["lr"] != 1e-5
        or config["data"]["buckets"] != [[480, 640], [640, 480], [400, 640]]
        or config["data"]["rgb_replacement"]["seed"] != 43
        or config["data"]["rgb_replacement"]["manifest"] != plan["synthetic_manifest"]
    ):
        raise ValueError("Wrong frozen Muon continuation policy")
    check_continued_counts(plan["source_optimizer_counts"], plan["source_optimizer_counts"], 0)


def check_performance(config, parent, plan, performance):
    check_contract(config, parent, plan)
    expected = dict(
        authorized=True,
        effective_batch=BATCH,
        compute_batch=BATCH,
        workers=4,
        data_pipeline="prepared",
        compact_transport=True,
        gpu_prefetch=True,
        anchor_cursor=0,
        anchor_step=parent["step"],
        anchor_schedule_position=float(parent["step"]),
    )
    if (
        any(performance.get(k) != v for k, v in expected.items())
        or performance.get("autotune")
        or performance.get("data_autotune")
    ):
        raise ValueError("Continuation requires the existing actual128 prepared pipeline")


class MuonContinuationAudit(OptimizerArmAudit):
    preserves_core_optimizer = True

    def __init__(self, root, plan, config, parent, progress):
        super().__init__(root, plan, config, parent, progress)
        self.updates = UPDATES

    def restore_optimizer(self, optimizer, state):
        optimizer.load_state_dict(state["optimizer"])
        self.check_optimizer_counts(optimizer_counts(optimizer.state_dict()), 0)

    def check_optimizer_counts(self, counts, updates):
        check_continued_counts(counts, self.plan["source_optimizer_counts"], updates)

    def check_batch_transition(self, config, parent, performance):
        check_performance(config, parent, self.plan, performance)

    def check_fork(self, config, state, digests):
        check_contract(config, state, self.plan)
        if digests != state["data_digests"]:
            raise ValueError("Continuation changed training or evaluation identities")

    def load_records(self, path):
        if str(checked_path(self.plan["real_manifest"])) != path:
            raise ValueError("Original manifest changed")
        previous = checked_json(self.plan["source_metadata"])
        real = load_reference_records(path, self.progress, workers=4)
        synthetic = load_reference_records(checked_path(self.plan["synthetic_manifest"]), workers=4)
        check_rows(synthetic, "train")
        if (
            len(real) != 1597911
            or real.digest() != previous["real_digest"]
            or len(synthetic) != IMAGES
            or dict(Counter(r["dataset"] for r in synthetic)) != SOURCES
        ):
            raise ValueError("Frozen RGB population changed")
        for field, count in [("validation_manifest", 64), ("structural_validation_manifest", 317)]:
            heldout = read_manifest(checked_path(self.plan[field]))
            check_rows(heldout, "val")
            if len(heldout) != count:
                raise ValueError("Heldout count changed")
            check_split(synthetic, heldout)
        check_split(real, synthetic)
        self.records = ReplacementRecords(real, synthetic)
        if self.records.digest() != self.parent["data_digests"]["train"]:
            raise ValueError("Same-pool continuation changed its metadata digest")
        self.coverage = MultiEpochCoverage(len(real), synthetic)
        self.planned_keys = [
            list(k)
            for b in MultiEpochBatches(
                self.records,
                self.config["data"]["rgb_replacement"],
                4,
                self.config["data"]["buckets"],
                self.config["train"]["seed"],
                0,
                UPDATES * BATCH // 4,
            )
            for k in b
        ]
        epoch_checks = []
        for epoch in range(EPOCHS):
            keys = self.planned_keys[
                epoch * UPDATES_PER_EPOCH * BATCH : (epoch + 1) * UPDATES_PER_EPOCH * BATCH
            ]
            indices = [k[0] - len(real) for k in keys if k[0] >= len(real)]
            if sorted(indices) != list(range(IMAGES)):
                raise ValueError("A planned pass omitted or repeated an identity")
            epoch_checks.append(
                dict(
                    epoch=epoch + 1,
                    synthetic=len(indices),
                    unique=len(set(indices)),
                    repeated=0,
                    tail_synthetic=IMAGES % 16,
                )
            )
        publish_json(self.root / "planned-keys.json", self.planned_keys)
        publish_json(
            self.root / "metadata-report.json",
            dict(
                stage="three_full_passes_planned",
                real_images=len(real),
                synthetic_images=IMAGES,
                epochs=EPOCHS,
                real_digest=real.digest(),
                synthetic_digest=synthetic.digest(),
                mixed_digest=self.records.digest(),
                planned_source_counts=dict(
                    Counter(self.records[k[0]]["dataset"] for k in self.planned_keys)
                ),
                planned_keys=file_ref(self.root / "planned-keys.json"),
                planned_epochs=epoch_checks,
                planned_synthetic_unique=IMAGES,
                planned_synthetic_presentations=IMAGES * EPOCHS,
                repeats_only_across_complete_epochs=True,
                real_start_cursor=real_cursor(self.parent),
                parent_mixed_cursor=self.parent["cursor"],
                bucket_policy=BUCKET_POLICY,
                geometry_loss=False,
                new_epoch_permutation_seeds=[43, 44, 45],
            ),
        )
        return self.records

    def finish(self, step, cursor):
        result = RGBInputAudit.finish(self, step, cursor)
        coverage = self.coverage.report()
        if self.committed == self.updates and not coverage["every_image_once_per_epoch"]:
            raise ValueError("Final three-pass coverage incomplete")
        publish_json(self.root / "coverage-counts.json", self.coverage.counts)
        publish_json(
            self.root / "coverage-final.json",
            dict(
                stage="committed_three_pass_coverage_verified",
                step=step,
                updates=self.committed,
                **coverage,
                counts=file_ref(self.root / "coverage-counts.json"),
                actual_inputs=file_ref(self.path),
                unused_prefetch_excluded=True,
                matched_encoded_image_identity=True,
                budget_completed=self.committed == self.updates,
            ),
        )
        return result


def run(plan_path):
    root = plan_path.parent
    plan = checked_json(file_ref(plan_path))
    with Path(plan["training_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "status.json").exists() or (root / "training").exists():
            raise FileExistsError("Already launched; inspect the existing process")
        status = dict(
            stage="checking_muon_continuation", started_at=time.time(), optimizer_updates=0
        )

        def progress(rows=0, stage="reading_verified_real_metadata"):
            status.update(stage=stage, metadata_rows=rows, updated_at=time.time())
            atomic_json(status, root / "status.json")

        progress(stage=status["stage"])
        try:
            config = checked_json(plan["config"])
            source = checked_path(plan["source_checkpoint"])
            parent = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
            check_contract(config, parent, plan)
            finish = checked_json(plan["source_finish"])
            if (
                finish["checkpoint"]["sha256"] != plan["source_checkpoint"]["sha256"]
                or finish["step"] != parent["step"]
                or optimizer_counts(parent["optimizer"]) != plan["source_optimizer_counts"]
            ):
                raise ValueError("Parent does not match the closed hybrid checkpoint")
            code, prior = Path(__file__).resolve().parents[1], Path(plan["source_execution"])
            if checked_path(plan["runner"]) != Path(__file__).resolve():
                raise ValueError("Frozen runner changed")
            for ref in plan["implementation_files"]:
                checked_path(ref)
            hashes = {}
            for name in (
                "data.py",
                "losses.py",
                "system.py",
                "performance.py",
                "prefetch.py",
                "evaluate.py",
                "decoder.py",
                "adapters.py",
                "architecture_screen.py",
                "attention_tricks.py",
                "feature_refinement.py",
                "rgb_replacement.py",
                "rgb_input_audit.py",
                "optimizer_trial.py",
            ):
                hashes[name] = sha_file(code / "src/radio_moge" / name)
                if hashes[name] != sha_file(prior / "src/radio_moge" / name):
                    raise ValueError("Unexpected feature/optimizer rule change: " + name)
            dependency = checked_json(plan["dependency"])
            for ref in dependency["python_files"]:
                checked_path(ref)
            import dion

            if (
                dependency["revision"] != config["optimizer_trial"]["dion_revision"]
                or Path(dion.__file__).resolve() != checked_path(dependency["package_init"])
                or Path(trainer.__file__).resolve() != checked_path(plan["trainer"])
            ):
                raise ValueError("Imported dependency or trainer differs")
            if config["output_dir"] != str(root / "training"):
                raise ValueError("A new output directory is required")
            publish_json(
                root / "source-module-identity.json",
                dict(
                    hashes=hashes,
                    feature_and_optimizer_rules_identical=True,
                    implementation_files=plan["implementation_files"],
                    dependency=plan["dependency"],
                    trainer=plan["trainer"],
                ),
            )
            performance = deepcopy(parent["performance"])
            performance.update(
                anchor_step=parent["step"],
                anchor_cursor=0,
                anchor_schedule_position=float(parent["step"]),
                original_started_at=time.time(),
                data_policy="Three full frozen synthetic passes; original real cursor continues",
                lr_policy="Preserve all Muon momentum and84 AdamW states; constant1e-5",
            )
            check_performance(config, parent, plan, performance)
            audit = MuonContinuationAudit(root, plan, config, parent, progress)
            save = trainer.atomic_save

            def checked_save(value, path):
                if Path(path).name == "initial.pt":
                    equal = initial_state_check(value, parent, config, performance)
                    save(value, path)
                    publish_json(
                        root / "initial-verification.json",
                        dict(
                            stage="full_hybrid_optimizer_continuation_verified",
                            equal_sections=list(equal),
                            source=plan["source_checkpoint"],
                            step=value["step"],
                            cursor=0,
                            parent_mixed_cursor=parent["cursor"],
                            real_stream_start_cursor=real_cursor(parent),
                            optimizer_counts=optimizer_counts(value["optimizer"]),
                            parameter_count=100,
                            core_optimizer=config["optimizer_trial"],
                            optimizer_updates=0,
                            partition=file_ref(root / "parameter-partition.json"),
                            learning_rate=1e-5,
                            all_optimizer_momenta_and_counters_bitwise=True,
                            core_state_reset=False,
                            scheduler=value["scheduler"],
                            data_digests=value["data_digests"],
                            performance=performance,
                        ),
                    )
                else:
                    save(value, path)

            torch.cuda.set_per_process_memory_fraction(0.88)
            with (
                patch.object(trainer, "atomic_save", checked_save),
                patch.object(trainer, "ReplacementBatches", MultiEpochBatches),
            ):
                final = trainer.train(
                    config,
                    fork_from=source,
                    data_fork=True,
                    batch_fork=True,
                    performance=performance,
                    reference=audit,
                    execution_revision=plan["execution_revision"],
                )
            stop = json.loads((root / "training/stop-report.json").read_text())
            coverage = checked_json(file_ref(root / "coverage-final.json"))
            complete = (
                stop["step"] == parent["step"] + UPDATES and coverage["every_image_once_per_epoch"]
            )
            publish_json(
                root / "completed.json",
                dict(
                    stage="muon_three_passes_finished" if complete else "incomplete",
                    budget_complete=complete,
                    full_epoch_complete=complete,
                    new_updates=stop["step"] - parent["step"],
                    final_checkpoint=file_ref(final),
                    stop_report=stop,
                    coverage=file_ref(root / "coverage-final.json"),
                    source=plan["source_checkpoint"],
                    automatic_next_segment=False,
                    quality_review_pending=True,
                ),
            )
            status.update(optimizer_updates=stop["step"] - parent["step"], final_step=stop["step"])
            progress(stage="finished_review_pending" if complete else "incomplete_inspect_no_retry")
        except Exception as error:
            status.update(error=type(error).__name__, message=str(error))
            progress(stage="failed_inspect_no_retry")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    torch.set_num_threads(1)
    run(parser.parse_args().plan.resolve())
