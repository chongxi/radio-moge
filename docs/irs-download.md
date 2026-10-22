# Download IRS

The [official HKBU-HPML/IRS repository](https://github.com/HKBU-HPML/IRS)
links to a public OneDrive folder. `scripts/download_irs.py` follows the current
anonymous OneDrive web flow, obtains a fresh file URL, and downloads one archive
at a time. It needs Python 3.11+ and `requests`; no browser, Microsoft account,
copied cookies, or manually supplied access token is required for this public share.

```bash
# Inspect the current source inventory; no dataset download.
uv run --script scripts/download_irs.py --list

# Verify real download access and nonzero-offset Range support (2 KiB per file).
uv run --script scripts/download_irs.py --probe --files Store.tar.gz

# Download the six full-release packages, one at a time.
uv run --script scripts/download_irs.py --download \
  --output /path/to/persistent-storage/irs-full

# Download only selected packages, or resume them with the same command.
uv run --script scripts/download_irs.py --download \
  --files Home-1.tar.gz Home-2.tar.gz \
  --output /path/to/persistent-storage/irs-full
```

An existing environment with `requests` can use `python scripts/download_irs.py`
instead of `uv run --script`. The inline dependency declaration keeps the `uv`
command independent of the project's PyTorch training dependencies.

The default selection is `Home-1`, `Home-2`, `Office-1`, `Office-2`, `Restaurant`,
and `Store` (`.tar.gz`). On 2026-10-22 these total **711,038,354,311 bytes**
(711.0 GB / 662.2 GiB), before extraction. `IRS_small.tar.gz` is an overlapping
subset and is excluded by default. Select it explicitly with `--files
IRS_small.tar.gz`, or add `--include-small` to download it alongside the full set.
Always inspect the live inventory: source sizes and versions can change.

Downloads use `.part` files and identity sidecars. Rerunning the command resumes
from the existing byte count only if the source ID, size, and ETag still match.
The server must return the exact requested `Content-Range` and ETag; a server
that ignores Range cannot silently corrupt an existing partial file. Transient
failures retry up to five times by default (`--retries N`), obtaining a fresh
anonymous session and signed file URL on each attempt. Version changes stop the
download for inspection. Do not delete or edit sidecars to force a resume.

After transfer, the script streams the entire gzip archive to verify its CRC
and computes the compressed file's SHA256. This takes CPU and disk I/O; progress
enters `verifying_gzip`. The SHA256 is a local integrity record, not a claim of
matching an unpublished publisher checksum. Verified archives and receipts are
written under `archives/` and `receipts/`; `inventory.json` and `status.json`
contain public metadata. Temporary cookies, bearer tokens, and signed download
URLs remain in memory and are not included in logs or receipts.

The script does not extract files, construct train/validation splits, or launch
training. Allow additional space for extraction. The official train/test lists
share scene directories, so they should not be treated as a scene-disjoint
generalization benchmark without further preparation. Native disparity and
normal labels are not automatically enabled as training targets.

For this project's current setup, bulk datasets belong on the GPU machine at
`/home/buildhub/shared/radio-moge/datasets/`, not the small VS Code workspace
disk. `/shared/radio-moge/datasets/` in the CPU workspace holds separately copied
backups. Run only one bulk dataset download/copy at a time. This script's lock
prevents duplicate downloads into the same IRS output directory; it does not
coordinate other dataset jobs or automatically join their queue.

The OneDrive web endpoints can change. If listing fails, inspect the official
share in a browser and update the web-flow implementation; a failure of the
legacy `api.onedrive.com/v1.0/shares` API alone does not show that the files are
unavailable. `--share-url` can select a replacement official public folder.
