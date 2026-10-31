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

## Authorized recovery, 2026-10-30

The original downloader stopped with `BrokenPipeError`; the user subsequently
requested continuation. The wrapper now accepts `--resume --resume-id ID` while
holding the original exclusive lock. It requires the stopped state and a new
recovery ID, preserves the previous status/logs and all existing receipts, verifies
frozen metadata/selection, receipt totals, retained raw-file sizes, published asset
paths and ZIP CRC/extraction receipts, and validates the interrupted partial's
source validator before resuming. This does not rehash the retained dataset.

A completed ZIP is normally removed by the official script after extraction; its
transfer and extraction receipts remain authoritative for the prior download.
Completed metadata/raw transfers are never downloaded or counted again. Unverified
existing files, missing extracted assets, changed metadata and invalid partial
identities cause a stop, not a replacement. The original official script still
chooses the remaining assets, and final completion also requires full receipt
coverage. New error records contain sanitized stack locations without source lines.

The read-only recovery preflight found 14,307 transfer receipts: 14,295 dataset
assets plus 12 metadata files, with 12,387 CRC-checked extractions. The total of
943,459,341,973 transferred bytes includes metadata. A 94,224,067-byte interrupted
partial was preserved. File-count progress must distinguish asset and metadata
receipts; neither is a byte-based completion percentage. Each new recovery still
requires an explicit request; the wrapper does not automatically restart itself.

## Uncapped download requested after training, 2026-10-30

The user explicitly requested maximum download speed after all training finished.
The 20 MiB/s cap was local throttling, not a measured publisher/network limit.
`--rate-limit-mib-s 0 --nice 0` removes bandwidth sleeps and restores normal CPU
priority for the downloader. Content-Length, source validators, streamed SHA,
ZIP CRC, staging and receipt-preserving recovery remain enabled. The only active
controller is replaced once at an incomplete network transfer, with the partial
preserved and its receipt/identity checks repeated before continuation. It is not
interrupted during ZIP extraction or publication.

New transfer receipts record offset, transferred bytes, elapsed time and intentional
sleep time. Report actual sustained throughput, including extraction gaps when
estimating total completion; an unlimited setting does not prove unlimited network
capacity. No training jobs are started and no private official file is modified.

Uncapped single-connection transfer of the interrupted 334,547,892-byte video
measured 324,062,132 new bytes in 19.2455 seconds (about16.06MiB/s), with zero
intentional sleep. Large uncapped transfers can therefore use `--range-workers 4`
(up to8). Each16MiB range requires exact Content-Range/Length and unchanged server
validators. Completed ranges are assembled in byte order into the existing partial
and full-file SHA. Memory and concurrent requests are bounded; small files use one
connection. If the first range is ignored, the untouched original response is used
serially. Later range errors stop and preserve the valid prefix.

An optional public `--performance-file` containing only `range_workers` allows
changing between1and8 at the next asset without another restart. It contains no
URLs or credentials. Higher concurrency is evaluated with real remaining assets;
no additional full-dataset reads or training are involved. New receipts distinguish
requested and actual range workers.

## Shorter parallel-range probe, 2026-10-31

A read-only sample of 167 completed transfers since the latest resume covered
1,463 seconds: transfer-body timing accounted for879 seconds, completed ZIP
extraction/publication gaps for394 seconds, and the remaining190 seconds included
startup, requests, metadata and the current unfinished operation. Large parallel
transfers averaged19.81MB/s within their recorded transfer intervals, while overall
verified-byte progress was10.35MB/s. These observations do not establish a network
limit or attribute all waiting to any one system component.

The previous parallel path first fetched16MiB through a single request before
starting its worker pool. It now validates a64KiB prefix (or a smaller configured
chunk) and immediately schedules the remaining16MiB ranges. Every range retains
the original source-validator, Content-Range and length checks; SHA assembly,
fallback, worker bounds and extraction behavior are unchanged. Receipts additionally
record probe bytes and duration. Tests cover nonzero resume offsets, contiguous
coverage and invalid later ranges. This removes most serial probe bytes; sustained
throughput improvement must be measured after deployment, not inferred from the
smaller probe alone. Download and extraction are still sequential.
