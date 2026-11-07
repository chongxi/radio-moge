"""One bounded core-reset AdamW or Muon arm on the exact saved LR reference stream."""

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
from functools import partial
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from radio_moge.checkpoint import atomic_json, rng_state, restore_rng
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.corpus_model_probe import assert_frozen, gradient_summary, module_digest
from radio_moge.data import TrainingImages, check_split, read_manifest
from radio_moge.geometry_cache import publish_json, sha_file
from radio_moge.losses import weighted_loss
from radio_moge.prefetch import normalize_uint8
from radio_moge.rgb_input_audit import RGBInputAudit, pixels_sha, load_reference_records
from radio_moge.rgb_replacement import ReplacementRecords
import radio_moge.train as trainer
from prepare_room_rgb_pool import check_rows
from run_room_feature_epoch import real_cursor, IMAGES, SOURCES
from verify_corpus_stage_startup import equal_finite
from radio_moge.optimizer_trial import (
    CoreOptimizerTrial,
    policy,
    optimizer_counts,
    check_counts,
    MODES,
)
import run_room_lr_arm as reference_arm
from run_room_true_batch_epoch import TrueBatchSampler, BatchCoverage, BUCKET_POLICY

SECONDS = 5400
UPDATES = 1000
BATCH = 128
SYNTHETIC_PRESENTATIONS = UPDATES * BATCH // 8
COVERAGE_GOAL = "matched_fixed_1000_update_prefix_not_full_epoch"


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = type(value)((k, cpu(v)) for k, v in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)
        return result
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def initial_state_check(value, parent, config, performance):
    if (
        value["step"] != parent["step"]
        or value["cursor"] != 0
        or value["config"] != config
        or value["performance"] != performance
    ):
        raise ValueError("Actual initial cursor/configuration differs")
    sections = ("adapter", "statistics", "rng", "adapter_config", "model_config", "revisions")
    for name in sections:
        if not equal_finite(cpu(value[name]), parent[name]):
            raise ValueError("Actual initial state differs: " + name)
    current = value["optimizer"]
    proof = current["optimizer_trial"]
    if proof["policy"] != config["optimizer_trial"] or len(proof["names"]) != 100:
        raise ValueError("Actual named optimizer partition differs")
    ids = current["param_groups"][0]["params"]
    by_name = dict(zip(proof["names"], ids, strict=True))
    if current["param_groups"] != parent["optimizer"]["param_groups"]:
        raise ValueError("Original AdamW options changed")
    for name in proof["auxiliary_names"]:
        index = by_name[name]
        if not equal_finite(cpu(current["state"][index]), parent["optimizer"]["state"][index]):
            raise ValueError("An auxiliary AdamW state changed: " + name)
    if any(by_name[n] in current["state"] for n in proof["core_names"]):
        raise ValueError("Core Adam states were not reset")
    if "muon" in current:
        for row in current["muon"]["state"].values():
            if set(row) != {"momentum"} or torch.count_nonzero(row["momentum"]):
                raise ValueError("Muon momentum was not initialized to zero")
    check_counts(optimizer_counts(current), config["optimizer_trial"]["mode"], parent["step"], 0)
    scheduler = value["scheduler"]
    if (
        scheduler["base_lrs"] != [config["train"]["lr"]]
        or scheduler["_last_lr"] != scheduler["base_lrs"]
        or scheduler["last_epoch"] != 0
        or scheduler["_step_count"] != 1
    ):
        raise ValueError("Constant scheduler position changed")
    return sections


def canonical_config(config):
    result = deepcopy(config)
    result.pop("output_dir", None)
    result.pop("tracking", None)
    result.pop("optimizer_trial", None)
    return result


def compare_arm_configs(a, b):
    if canonical_config(a) != canonical_config(b):
        raise ValueError("Matched arms differ beyond core optimizer, output and tracking")


def normalized(config, plan):
    cfg = deepcopy(config)
    trial = cfg.pop("optimizer_trial", None)
    if plan.get("arm") not in MODES or trial != policy(plan["arm"]):
        raise ValueError("Wrong explicitly frozen optimizer policy")
    if plan["learning_rate"] != 1e-5 or plan["reference_arm"] is not None:
        raise ValueError(
            "Only the baseline LR and separately bound historical reference are allowed"
        )
    mapped = dict(plan, arm="lr1e-5")
    return cfg, mapped


def check_contract(config, parent, plan):
    cfg, mapped = normalized(config, plan)
    reference_arm.check_contract(cfg, parent, mapped)


def check_performance(config, parent, plan, performance):
    cfg, mapped = normalized(config, plan)
    reference_arm.check_performance(cfg, parent, mapped, performance)


class OptimizerArmAudit(RGBInputAudit):
    def __init__(self, root, plan, config, parent, progress):
        self.root, self.plan, self.config, self.parent, self.progress = (
            root,
            plan,
            config,
            parent,
            progress,
        )
        self.batch = plan["actual_batch"]
        self.updates = UPDATES
        super().__init__(config["output_dir"], parent["step"], 4, self.batch // 4, self.batch)

    def make_optimizer(self, adapter, settings):
        self.optimizer = CoreOptimizerTrial(
            adapter.named_parameters(), settings, self.config["optimizer_trial"]
        )
        identity = self.optimizer.identity()
        if len(identity["names"]) != 100 or len(identity["core_names"]) != 16:
            raise ValueError("Actual adapter partition differs from the frozen 16/84 plan")
        publish_json(self.root / "parameter-partition.json", identity)
        return self.optimizer

    def restore_optimizer(self, optimizer, state):
        optimizer.load_parent_adam(state["optimizer"])

    def check_batch_transition(self, config, parent, performance):
        check_performance(config, parent, self.plan, performance)

    def check_fork(self, config, state, digests):
        check_contract(config, state, self.plan)
        if digests != state["data_digests"]:
            raise ValueError("Batch change must preserve every training and heldout identity")

    def load_records(self, path):
        if str(checked_path(self.plan["real_manifest"])) != path:
            raise ValueError("Original manifest changed")
        previous = checked_json(self.plan["source_metadata"])
        if previous["mixed_digest"] != self.parent["data_digests"]["train"]:
            raise ValueError("Parent stream metadata mismatch")
        real = load_reference_records(path, self.progress, workers=4)
        synthetic = load_reference_records(checked_path(self.plan["synthetic_manifest"]), workers=4)
        check_rows(synthetic, "train")
        if (
            len(real) != 1597911
            or real.digest() != previous["real_digest"]
            or len(synthetic) != IMAGES
            or dict(Counter(r["dataset"] for r in synthetic)) != SOURCES
        ):
            raise ValueError("Frozen population or metadata changed")
        for field, count in [("validation_manifest", 64), ("structural_validation_manifest", 317)]:
            heldout = read_manifest(checked_path(self.plan[field]))
            check_rows(heldout, "val")
            if len(heldout) != count:
                raise ValueError("Validation population changed")
            check_split(synthetic, heldout)
        check_split(real, synthetic)
        self.records = ReplacementRecords(real, synthetic)
        if self.records.digest() != self.parent["data_digests"]["train"]:
            raise ValueError("Batch-only fork changed the dataset digest")
        self.coverage = BatchCoverage(len(real), synthetic, self.batch)
        self.planned_keys = [
            list(k)
            for b in TrueBatchSampler(
                self.records,
                self.config["data"]["rgb_replacement"],
                4,
                self.config["data"]["buckets"],
                self.config["train"]["seed"],
                0,
                self.updates * self.batch // 4,
                actual_batch=self.batch,
            )
            for k in b
        ]
        indices = [k[0] - len(real) for k in self.planned_keys if k[0] >= len(real)]
        if (
            len(indices) != SYNTHETIC_PRESENTATIONS
            or len(set(indices)) != len(indices)
            or any(not 0 <= i < IMAGES for i in indices)
        ):
            raise ValueError("Matched prefix omitted/repeated a planned identity")
        publish_json(self.root / "planned-keys.json", self.planned_keys)
        counts = Counter(self.records[k[0]]["dataset"] for k in self.planned_keys)
        publish_json(
            self.root / "metadata-report.json",
            dict(
                stage="matched_optimizer_prefix_plan_verified",
                real_images=len(real),
                synthetic_images=IMAGES,
                real_digest=real.digest(),
                synthetic_digest=synthetic.digest(),
                mixed_digest=self.records.digest(),
                planned_source_counts=dict(counts),
                planned_keys=file_ref(self.root / "planned-keys.json"),
                planned_synthetic_unique=SYNTHETIC_PRESENTATIONS,
                planned_synthetic_repeats=0,
                real_start_cursor=real_cursor(self.parent),
                parent_mixed_cursor=self.parent["cursor"],
                image_identities_and_crop_seeds_preserved=True,
                resolution_assignment_changed=False,
                bucket_policy=BUCKET_POLICY,
                full_batch_images=self.batch,
                synthetic_images_per_update=self.batch // 8,
                last_update_synthetic_images=self.batch // 8,
                geometry_loss=False,
            ),
        )
        other = Path(self.plan["baseline_arm"])
        other_plan = checked_json(self.plan["baseline_plan"])
        compare_arm_configs(self.config, checked_json(other_plan["config"]))
        for field in (
            "source_checkpoint",
            "source_step",
            "source_cursor",
            "source_parent_cursor",
            "synthetic_manifest",
            "real_manifest",
            "source_results",
            "bucket_policy",
        ):
            if self.plan[field] != other_plan[field]:
                raise ValueError("Historical matched reference differs: " + field)
        if (
            file_ref(self.root / "planned-keys.json")["sha256"]
            != file_ref(other / "planned-keys.json")["sha256"]
        ):
            raise ValueError("Planned input keys differ from the completed reference")
        return self.records

    def before_updates(self, system, adapter, statistics, records, config, output):
        self.progress(len(records), stage="checking_actual_full_batch_model_before_updates")
        modules = dict(
            radio=system.radio,
            encoder=system.encoder,
            decoder=system.decoder,
            adapter=adapter,
            statistics=statistics,
        )
        before = {k: module_digest(v) for k, v in modules.items()}
        rng = rng_state()
        data = TrainingImages(records, config["data"]["augment"], compact_transport=True)
        rows = []
        initial_optimizer = cpu(self.optimizer.state_dict())
        initial_weights = cpu(adapter.state_dict())
        try:
            # Actual first and last complete 128-image batches of this fixed comparison budget.
            for label, keys in [
                ("first", self.planned_keys[: self.batch]),
                ("tail", self.planned_keys[-self.batch :]),
            ]:
                adapter.zero_grad(set_to_none=True)
                compact = torch.stack([data[k] for k in keys])
                image = normalize_uint8(compact).to(next(adapter.parameters()).device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    features, target = system.encode(image, adapter.indices)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = adapter(features)
                losses = statistics.loss(predicted, target)
                loss = weighted_loss(losses, config["loss"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite actual full-batch feature loss")
                loss.backward()
                gradients = gradient_summary(adapter.parameters())
                assert_frozen(system)
                norm = torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(), config["train"]["clip_grad"]
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite clipped gate gradients")
                torch.cuda.synchronize()
                started = time.perf_counter()
                self.optimizer.step()
                torch.cuda.synchronize()
                update_seconds = time.perf_counter() - started
                norms = {}
                for label_group, named in [
                    ("core", self.optimizer.core),
                    ("auxiliary", self.optimizer.auxiliary),
                ]:
                    squared, old_squared = 0.0, 0.0
                    for name, parameter in named:
                        actual = parameter.detach().cpu()
                        if not torch.isfinite(actual).all():
                            raise FloatingPointError("Nonfinite actual optimizer output: " + name)
                        squared += float((actual - initial_weights[name]).float().square().sum())
                        old_squared += float(initial_weights[name].float().square().sum())
                    norms[label_group] = dict(
                        l2=squared**0.5, relative_l2=(squared / max(old_squared, 1e-30)) ** 0.5
                    )
                    if squared <= 0:
                        raise ValueError("An optimizer partition did not update")
                counts = optimizer_counts(self.optimizer.state_dict())
                check_counts(counts, self.config["optimizer_trial"]["mode"], self.parent["step"], 1)
                if not equal_finite(
                    cpu(self.optimizer.state_dict()), cpu(self.optimizer.state_dict())
                ):
                    raise FloatingPointError("Nonfinite optimizer state")
                adapter.load_state_dict(initial_weights)
                self.optimizer.load_state_dict(initial_optimizer)
                check_counts(
                    optimizer_counts(self.optimizer.state_dict()),
                    self.config["optimizer_trial"]["mode"],
                    self.parent["step"],
                    0,
                )
                if not equal_finite(cpu(self.optimizer.state_dict()), initial_optimizer):
                    raise ValueError("Temporary optimizer gate did not restore all state")
                rows.append(
                    dict(
                        label=label,
                        shape=list(image.shape),
                        pixel_sha256=pixels_sha(compact),
                        keys=keys,
                        total=float(loss.detach()),
                        gradient=gradients,
                        temporary_optimizer_update_seconds=update_seconds,
                        temporary_update_norms=norms,
                        optimizer_counts_after_temporary_step=counts,
                        peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                )
                del image, compact, features, target, predicted, losses, loss
            after = {k: module_digest(v) for k, v in modules.items()}
            if before != after:
                raise ValueError("Zero-update true batch gate modified model state")
        finally:
            adapter.load_state_dict(initial_weights)
            self.optimizer.load_state_dict(initial_optimizer)
            check_counts(
                optimizer_counts(self.optimizer.state_dict()),
                self.config["optimizer_trial"]["mode"],
                self.parent["step"],
                0,
            )
            adapter.zero_grad(set_to_none=True)
            restore_rng(rng)
        publish_json(
            self.root / "mixed-model-zero-update.json",
            dict(
                stage="actual_full_batch_model_passed",
                rows=rows,
                actual_batch=self.batch,
                cases=len(rows),
                images=self.batch * len(rows),
                state_before=before,
                state_after=after,
                optimizer_updates=0,
                gradients_discarded=True,
                temporary_optimizer_updates=2,
                temporary_updates_discarded_and_state_restored=True,
                current_optimizer=config["optimizer_trial"],
                rng_restored_bitwise=True,
                geometry_branch_executed=False,
                prior_all_three_resolution_probe=self.plan["batch_probe"],
            ),
        )
        if self.plan.get("baseline_arm"):
            previous = checked_json(
                file_ref(Path(self.plan["baseline_arm"]) / "mixed-model-zero-update.json")
            )
            if previous["state_before"] != before:
                raise ValueError("Actual paired model initial states differ")
            for left, right in zip(previous["rows"], rows, strict=True):
                if any(
                    left[k] != right[k] for k in ("label", "shape", "pixel_sha256", "keys", "total")
                ):
                    raise ValueError("Actual paired pre-update pixels or feature losses differ")
        baseline = checked_json(self.plan["source_results"])
        if baseline["step"] != self.parent["step"] or len(baseline["views"]) != 6:
            raise ValueError("Wrong saved six-view baseline")
        for view in baseline["views"].values():
            for ref in view["files"].values():
                checked_path(ref)
        publish_json(
            self.root / "baseline-reuse.json",
            dict(
                stage="six_saved_parent_views_bound",
                source=self.plan["source_results"],
                step=self.parent["step"],
                views={k: v["files"] for k, v in baseline["views"].items()},
                model_inference=False,
            ),
        )
        self.progress(len(records), stage="training_matched_optimizer_arm")

    def observe_window(self, tensors):
        if len(tensors) != 1 or len(tensors[0]) != self.batch:
            raise ValueError("Prepared window was split into multiple forward batches")
        super().observe_window(tensors)

    def commit(self, step, cursor):
        with self.lock:
            rows = self.pending_windows[0]["logical_images"]
        indices = self.coverage.validate(rows)
        super().commit(step, cursor)
        self.coverage.commit(indices)
        if self.committed % 100 == 0 or self.committed == self.updates:
            atomic_json(
                dict(step=step, updates=self.committed, **self.coverage.report()),
                self.root / "coverage-progress.json",
            )

    def finish(self, step, cursor):
        result = super().finish(step, cursor)
        coverage = self.coverage.report()
        if self.committed == self.updates and (
            coverage["unique"] != SYNTHETIC_PRESENTATIONS
            or coverage["committed"] != SYNTHETIC_PRESENTATIONS
            or coverage["repeated"]
        ):
            raise ValueError("Incomplete fixed-budget synthetic prefix")
        publish_json(self.root / "coverage-counts.json", self.coverage.counts)
        publish_json(
            self.root / "coverage-final.json",
            dict(
                stage="committed_synthetic_coverage_verified",
                step=step,
                updates=self.committed,
                **coverage,
                counts=file_ref(self.root / "coverage-counts.json"),
                actual_inputs=file_ref(self.path),
                unused_prefetch_excluded=True,
                matched_encoded_image_identity=True,
                budget_completed=self.committed == self.updates,
                full_epoch_completed=False,
            ),
        )
        return result


def run(plan_path):
    root = plan_path.parent
    plan = checked_json(file_ref(plan_path))
    with Path(plan["training_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "status.json").exists() or (root / "training").exists():
            raise FileExistsError("Already launched; inspect existing process")
        status = dict(
            stage="checking_true_batch_configuration", started_at=time.time(), optimizer_updates=0
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
            ):
                raise ValueError("Parent is not the verified final checkpoint")
            # The same source architecture/runtime already completed a full actual128
            # epoch. Reuse its performance evidence as historical evidence, then run
            # the current-checkpoint zero-update first/last model gate below.
            previous_gate = checked_json(plan["source_model_gate"])
            if (
                previous_gate["stage"] != "actual_full_batch_model_passed"
                or previous_gate["actual_batch"] != BATCH
                or previous_gate["state_before"] != previous_gate["state_after"]
                or finish["step"] != plan["source_step"]
                or finish["actual_rgb_presentations"] != 406400
            ):
                raise ValueError("Missing completed same-runtime actual128 source proof")
            if config["output_dir"] != str(root / "training"):
                raise ValueError("A separate training destination is required")
            code, prior = Path(__file__).resolve().parents[1], Path(plan["source_execution"])
            if checked_path(plan["runner"]) != Path(__file__).resolve():
                raise ValueError("Frozen batch runner changed")
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
            ):
                if sha_file(code / "src/radio_moge" / name) != sha_file(
                    prior / "src/radio_moge" / name
                ):
                    raise ValueError("Unexpected core change: " + name)
                hashes[name] = sha_file(code / "src/radio_moge" / name)
            transition = checked_json(plan["trainer_patch"])
            if transition["before_sha256"] != sha_file(
                prior / "src/radio_moge/train.py"
            ) or transition["after_sha256"] != sha_file(code / "src/radio_moge/train.py"):
                raise ValueError("Declared trainer-only optimizer transition is stale")
            checked_path(transition["patch"])
            checked_path(plan["optimizer_implementation"])
            dependency = checked_json(plan["dependency"])
            if dependency["revision"] != config["optimizer_trial"]["dion_revision"]:
                raise ValueError("Dion revision differs from the frozen policy")
            for ref in dependency["python_files"]:
                checked_path(ref)
            import dion

            if Path(dion.__file__).resolve() != Path(dependency["package_init"]["path"]).resolve():
                raise ValueError("Unexpected imported Dion package")
            checked_path(plan["trainer"])
            if Path(plan["trainer"]["path"]) != code / "src/radio_moge/train.py":
                raise ValueError("The batch-fork trainer implementation differs")
            if Path(trainer.__file__).resolve() != checked_path(plan["trainer"]):
                raise ValueError("The imported trainer differs from the frozen execution")
            publish_json(
                root / "source-module-identity.json",
                dict(
                    hashes=hashes,
                    feature_forward_backward_modules_unchanged=True,
                    trainer=plan["trainer"],
                    trainer_patch=plan["trainer_patch"],
                    dependency=plan["dependency"],
                    optimizer_implementation=plan["optimizer_implementation"],
                ),
            )
            performance = deepcopy(parent["performance"])
            batch = plan["actual_batch"]
            performance.update(
                effective_batch=batch,
                compute_batch=batch,
                workers=4,
                gpu_prefetch=True,
                anchor_step=parent["step"],
                anchor_cursor=0,
                anchor_schedule_position=float(parent["step"]),
                original_started_at=time.time(),
                schedule_ratio=batch / 16,
                data_policy="Matched1000 updates; identical128 RGB keys/crops/buckets in both arms",
                lr_policy="Fresh core optimizer; preserve auxiliary AdamW/RNG/RMS; constant base LR1e-5",
            )
            check_performance(config, parent, plan, performance)
            audit = OptimizerArmAudit(root, plan, config, parent, progress)
            save = trainer.atomic_save

            def checked_save(value, path):
                if Path(path).name == "initial.pt":
                    equal = initial_state_check(value, parent, config, performance)
                    save(value, path)
                    publish_json(
                        root / "initial-verification.json",
                        dict(
                            stage="actual_scoped_optimizer_transition_verified_before_updates",
                            equal_sections=list(equal),
                            source=plan["source_checkpoint"],
                            step=value["step"],
                            cursor=0,
                            parent_mixed_cursor=parent["cursor"],
                            real_stream_start_cursor=real_cursor(parent),
                            optimizer_states=len(value["optimizer"]["state"]),
                            optimizer_counts=optimizer_counts(value["optimizer"]),
                            parameter_count=100,
                            core_optimizer=config["optimizer_trial"],
                            partition=file_ref(root / "parameter-partition.json"),
                            optimizer_updates=0,
                            explicit_lr_policy_fields=["lr", "initial_lr"],
                            learning_rate=config["train"]["lr"],
                            auxiliary_adam_moments_counters_and_group_options_bitwise=True,
                            all_core_optimizer_states_fresh=True,
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
                patch.object(
                    trainer, "ReplacementBatches", partial(TrueBatchSampler, actual_batch=batch)
                ),
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
            complete = stop["step"] == parent["step"] + UPDATES and coverage["budget_completed"]
            if complete:
                other_audit = checked_json(
                    file_ref(Path(plan["baseline_arm"]) / "training/actual-rgb-input-audit.json")
                )
                current_audit = checked_json(
                    file_ref(root / "training/actual-rgb-input-audit.json")
                )
                if current_audit["records"]["sha256"] != other_audit["records"]["sha256"]:
                    raise ValueError("Full paired actual input bytes differ")
            publish_json(
                root / "completed.json",
                dict(
                    stage="matched_optimizer_arm_finished" if complete else "incomplete",
                    budget_complete=complete,
                    full_epoch_complete=False,
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    torch.set_num_threads(1)
    run(p.parse_args().plan.resolve())
