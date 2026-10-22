#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests>=2.32,<3"]
# ///
"""Download IRS through its public OneDrive web share, with resumable transfers.

Default: list files only. Use --probe for small real range requests, or --download
--output PATH for archives. Temporary web credentials and signed URLs stay in RAM.
"""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

import requests


SHARE = "https://1drv.ms/f/s!AmN7U9URpGVGem0coY8PJMHYg0g?e=nvH5oB"
FULL = (
    "Home-1.tar.gz",
    "Home-2.tar.gz",
    "Office-1.tar.gz",
    "Office-2.tar.gz",
    "Restaurant.tar.gz",
    "Store.tar.gz",
)
CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    """Messages must not contain temporary URLs, cookies, or bearer tokens."""


class SourceChanged(DownloadError):
    pass


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def public_item(item):
    return {key: item[key] for key in ("name", "bytes", "id", "etag")}


def require_status(response, expected, operation):
    if response.status_code != expected:
        raise DownloadError(f"{operation}: HTTP {response.status_code}")


class IRSClient:
    def __init__(self, share_url=SHARE):
        self.share_url = share_url
        self.session = None
        self.items = {}
        self.token = None

    def refresh(self):
        """Follow the same anonymous SharePoint flow as the current OneDrive UI."""
        if self.session:
            self.session.close()
        self.session = requests.Session()
        landing = self.session.get(self.share_url, timeout=(20, 60))
        require_status(landing, 200, "Public share page")
        url = urlsplit(landing.url)
        query = parse_qs(url.query)
        if url.scheme != "https" or url.hostname != "onedrive.live.com":
            raise DownloadError("Share redirected outside the supported OneDrive web view")
        try:
            folder, library = query["spopath"][0], query["listurl"][0]
        except KeyError:
            raise DownloadError(
                "OneDrive did not provide spopath/listurl; web flow may have changed"
            ) from None
        if not library.startswith("/personal/") or not folder.startswith(library + "/"):
            raise DownloadError("Unexpected OneDrive folder/library relationship")
        site = "https://onedrive.live.com" + library.rsplit("/", 1)[0]
        context = self.session.post(
            site + "/_api/contextinfo",
            headers={"Accept": "application/json;odata=verbose"},
            timeout=(20, 60),
        )
        require_status(context, 200, "Anonymous web context")
        digest = context.json()["d"]["GetContextWebInformation"]["FormDigestValue"]
        response = self.session.post(
            site + "/_api/web/GetListUsingPath(DecodedUrl=@a1)/RenderListDataAsStream",
            params={
                "@a1": "'" + library.replace("'", "''") + "'",
                "RootFolder": folder,
                "TryNewExperienceSingle": "TRUE",
            },
            headers={
                "Accept": "application/json;odata=verbose",
                "Content-Type": "application/json;odata=verbose",
                "X-RequestDigest": digest,
            },
            json={
                "parameters": {
                    "__metadata": {"type": "SP.RenderListDataParameters"},
                    "RenderOptions": 5691175,
                    "AddRequiredFields": True,
                    "AllowMultipleValueFilterForTaxonomyFields": True,
                    "RequireFolderColoringFields": True,
                }
            },
            timeout=(20, 60),
        )
        require_status(response, 200, "Public archive listing")
        body = response.json()
        if body["ListData"].get("NextHref"):
            raise DownloadError(
                "Listing is paginated; refusing to treat a partial listing as complete"
            )
        self.token = parse_qs(body["ListSchema"][".driveAccessToken"])["access_token"][0]
        items = {}
        for row in body["ListData"]["Row"]:
            name = row["FileLeafRef"]
            if row["FSObjType"] != "0" or not name.endswith(".tar.gz"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz", name):
                raise DownloadError("Unsafe archive filename in source listing")
            endpoint = urlsplit(row[".spItemUrl"])
            if endpoint.scheme != "https" or endpoint.hostname != "onedrive.live.com":
                raise DownloadError("Unexpected item metadata host")
            if name in items:
                raise DownloadError("Duplicate archive name in source listing")
            items[name] = dict(
                name=name,
                bytes=int(row["File_x0020_Size"]),
                id=row["UniqueId"],
                etag=row[".etag"],
                item_url=row[".spItemUrl"],
            )
        self.items = items
        return items

    def resolve(self, expected):
        # A new anonymous session also refreshes expiring signed links on every retry.
        self.refresh()
        current = self.items.get(expected["name"])
        if current is None or public_item(current) != public_item(expected):
            raise SourceChanged("Archive identity changed; existing partial file preserved")
        response = self.session.get(
            current["item_url"], headers={"Authorization": "Bearer " + self.token}, timeout=(20, 60)
        )
        require_status(response, 200, "Archive metadata")
        metadata = response.json()
        if metadata["size"] != expected["bytes"] or metadata["eTag"] != expected["etag"]:
            raise SourceChanged("Archive size/version changed; existing partial file preserved")
        link = metadata.get("@content.downloadUrl") or metadata.get("@microsoft.graph.downloadUrl")
        if not link or urlsplit(link).scheme != "https":
            raise DownloadError("OneDrive did not return an HTTPS download URL")
        return link


def select_items(items, names=None, include_small=False):
    selected = names or list(FULL) + (["IRS_small.tar.gz"] if include_small else [])
    if len(set(selected)) != len(selected):
        raise DownloadError("Duplicate names in --files")
    missing = sorted(set(selected) - set(items))
    if missing:
        raise DownloadError("Missing requested archives: " + ", ".join(missing))
    return [public_item(items[name]) for name in selected]


def check_range(response, item, start, end):
    require_status(response, 206, "Range download (server must honor Range)")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if not match or tuple(map(int, match.groups())) != (start, end, item["bytes"]):
        raise DownloadError("Incorrect Content-Range; refusing to append to partial file")
    if response.headers.get("ETag") != item["etag"]:
        raise SourceChanged("Download ETag changed; existing partial file preserved")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise DownloadError("Unexpected HTTP content encoding on archive download")


def probe(client, item):
    link = client.resolve(item)
    results = []
    for start in (0, 1024):
        end = min(start + 1023, item["bytes"] - 1)
        if end < start:
            break
        with requests.get(
            link,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True,
            timeout=(20, 60),
        ) as response:
            check_range(response, item, start, end)
            data = response.raw.read(end - start + 2)
        if len(data) != end - start + 1 or (start == 0 and data[:2] != b"\x1f\x8b"):
            raise DownloadError("Range probe length or gzip header mismatch")
        results.append(
            dict(start=start, end=end, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        )
    return dict(name=item["name"], size=item["bytes"], range_probes=results, passed=True)


def verify_gzip(path, expected_size):
    if path.stat().st_size != expected_size:
        raise DownloadError("Archive size mismatch")
    digest = hashlib.sha256()

    class HashReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size=-1):
            block = self.stream.read(size)
            digest.update(block)
            return block

    unpacked = 0
    with path.open("rb") as stream:
        reader = HashReader(stream)
        with gzip.GzipFile(fileobj=reader, mode="rb") as archive:
            while block := archive.read(CHUNK):
                unpacked += len(block)
        while reader.read(CHUNK):
            pass
    return dict(
        sha256=digest.hexdigest(),
        bytes=expected_size,
        uncompressed_bytes=unpacked,
        verification="Full gzip decompression/CRC and local compressed-file SHA256; no extraction",
    )


def download_one(client, item, output, record, retries=5):
    archives = output / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    final = archives / item["name"]
    partial = archives / (item["name"] + ".part")
    identity = archives / (item["name"] + ".part.json")
    receipt = output / "receipts" / (item["name"] + ".json")
    if final.exists():
        if not receipt.exists():
            raise DownloadError("Existing final archive has no receipt; inspect before reusing")
        saved = json.loads(receipt.read_text())
        with final.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            saved["source"] != item
            or saved["sha256"] != sha
            or final.stat().st_size != item["bytes"]
        ):
            raise SourceChanged("Existing verified archive changed")
        return saved
    if identity.exists():
        if json.loads(identity.read_text()) != item:
            raise SourceChanged("Partial file belongs to another archive version")
    elif partial.exists():
        raise DownloadError("Partial file has no identity receipt; refusing unsafe resume")
    else:
        atomic_json(identity, item)
    if partial.exists() and partial.stat().st_size > item["bytes"]:
        raise DownloadError("Partial archive exceeds source size")
    for attempt in range(retries + 1):
        start = partial.stat().st_size if partial.exists() else 0
        if start == item["bytes"]:
            break
        try:
            link = client.resolve(item)
            end = item["bytes"] - 1
            with requests.get(
                link,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(20, 60),
            ) as response:
                check_range(response, item, start, end)
                received, last_log = 0, 0
                with partial.open("ab") as stream:
                    for block in response.iter_content(CHUNK):
                        if not block:
                            continue
                        if received + len(block) > end - start + 1:
                            raise DownloadError("Response exceeds declared byte range")
                        stream.write(block)
                        received += len(block)
                        if time.monotonic() - last_log >= 5:
                            stream.flush()
                            record(
                                stage="downloading",
                                archive=item["name"],
                                bytes=start + received,
                                total_bytes=item["bytes"],
                                retry=attempt,
                            )
                            last_log = time.monotonic()
                if received != end - start + 1:
                    raise DownloadError("Incomplete range response; partial preserved")
            break
        except SourceChanged:
            raise
        except (requests.RequestException, DownloadError) as error:
            if attempt == retries:
                raise DownloadError(
                    f"Transfer failed after {retries + 1} attempts ({type(error).__name__}); rerun to resume"
                ) from None
            record(
                stage="retrying",
                archive=item["name"],
                retry=attempt + 1,
                error_type=type(error).__name__,
            )
            time.sleep(min(30, 2**attempt))
    record(stage="verifying_gzip", archive=item["name"])
    result = dict(source=item, **verify_gzip(partial, item["bytes"]), verified_at=time.time())
    # Receipt precedes rename so a crash leaves a recoverable completed .part file.
    atomic_json(receipt, result)
    partial.replace(final)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List only (default)")
    mode.add_argument(
        "--probe", action="store_true", help="Read two 1 KiB ranges per selected file"
    )
    mode.add_argument(
        "--download", action="store_true", help="Download and verify selected archives"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--share-url", default=SHARE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--files", nargs="+", help="Exact archive names; default is six full packages"
    )
    selection.add_argument(
        "--include-small", action="store_true", help="Also download IRS_small (overlapping subset)"
    )
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.download and args.output is None:
        parser.error("--download requires --output")
    if args.retries < 0:
        parser.error("--retries must be nonnegative")
    status_path = None
    try:
        if args.download:
            args.output.mkdir(parents=True, exist_ok=True)
            lock = (args.output / ".download.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status_path = args.output / "status.json"
        client = IRSClient(args.share_url)
        items = client.refresh()
        selected = select_items(items, args.files, args.include_small)
        inventory = dict(
            checked_at=time.time(),
            archives=[public_item(x) for x in items.values()],
            selected=[x["name"] for x in selected],
            selected_bytes=sum(x["bytes"] for x in selected),
        )
        if not args.probe and not args.download:
            emit(inventory)
            return
        if args.probe:
            for item in selected:
                emit(probe(client, item))
            return
        atomic_json(args.output / "inventory.json", inventory)
        state = dict(pid=os.getpid(), started_at=time.time(), completed=[], training_ready=False)

        def record(**changes):
            state.update(changes, updated_at=time.time())
            atomic_json(status_path, state)
            emit(changes)

        for item in selected:
            download_one(client, item, args.output, record, args.retries)
            state["completed"].append(item["name"])
            record(stage="between_archives")
        record(stage="archives_complete", finished_at=time.time())
    except (Exception, KeyboardInterrupt) as error:
        # Requests exceptions may embed signed URLs. Never print their original text.
        message = str(error) if isinstance(error, DownloadError) else type(error).__name__
        report = dict(
            stage="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=message,
            updated_at=time.time(),
            pid=os.getpid(),
        )
        if status_path:
            atomic_json(status_path, report)
        emit(report)
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1) from None


if __name__ == "__main__":
    main()
