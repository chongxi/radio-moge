# ScanNet++ private download, 2026-10-28

Chongxi supplied the official download package and authorized downloading its default
assets after formal full-data training starts. The package and personal credential
must not be redistributed. They are deliberately outside this Git repository and
the project's shared artifacts.

## Locations and scope

- Private official scripts, original template and isolated download environment:
  `~/.local/share/radio-moge/download-scripts/scannetpp-v2-20261028/` on the GPU machine.
- The filled runtime YAML is mode `0600`; private directories are mode `0700`.
  Credentials never appear in process arguments, Git, shared reports or console output.
- Persistent dataset destination:
  `/home/buildhub/shared/radio-moge/datasets/scannetpp-v2-full` on the GPU machine.
- Sanitized plan, launch record and monitoring snapshots:
  `/shared/radio-moge/chongxi/data-expansion-attention-20261022/scannetpp-download-20261028`.

Use the supplied default selection: 30 asset kinds, all five configured splits,
and the official exclusions for each split. This template includes pano assets as
well as resized DSLR, iPhone, meshes and annotations. Optional original-resolution
DSLR and point-cloud asset groups are not enabled. The publisher's approximately
1.5 TB description is an estimate, not a verified byte total for this template.
The current server metadata resolves to 1,018 distinct scenes; record the actual
split counts and metadata hashes rather than substituting an older scene count.

The download is **not** permission to add these assets to the current training
manifests. Preserve the official train/validation/test partitions for later work.

## Execution and integrity

`scripts/run_private_dataset_download.py` wraps the unchanged private official
script. Its gate requires a real formal optimizer log row with a positive step,
learning rate and gradient, and finite numeric values. Preflight updates do not
open the gate. The existing full-data run opened it before download began.

One process runs at nice 15 with a 20 MiB/s network cap. Requirements are installed
in a private isolated venv; the training environment is untouched. Metadata and
asset selection remain the official script's responsibility. Before the first
asset, freeze the expected inventory and hashes of all downloaded split metadata.

Network writes go to `.part` files. Resume uses Range plus ETag/Last-Modified; a
server ignoring Range starts the transfer over instead of appending. Content-Length
must match before publication. Each completed transfer has a streamed SHA-256
receipt. These are local transfer checksums, **not publisher-provided checksums**.
ZIP extraction checks CRC and sizes in staging, rejects traversal, symlinks and
duplicate members, and then publishes the extracted entries. Final completion
requires the expected asset inventory as well as successful script return.

The original script can print the token or signed URLs on failures, so stdout and
stderr are buffered and redacted before they reach its private console log. The
monitor additionally checks that the credential is absent from the stored logs.
Only sanitized progress reaches the shared handoff and browser page.

Never start a second controller or blindly restart a failed download. Inspect the
private status and preserved partial files first. The token expires at
2026-11-05 09:14:43 UTC; no credential is included in this document.

## Monitoring

```sh
/usr/bin/python3 scripts/monitor_private_dataset_download.py \
  /shared/radio-moge/chongxi/data-expansion-attention-20261022/scannetpp-download-20261028
```

The active browser page is https://buildhub-nano-test--p7869.pub.astera.work/.
The existing server is reused. No additional GPU work or dataset copy to the local
workspace is part of this download.
