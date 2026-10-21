#!/usr/bin/env python3
"""Download the release data and restore the original 02_data layout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Streaming reads require a size")
        chunks = []
        remaining = size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            block = self.current.read(remaining)
            if block:
                chunks.append(block)
                remaining -= len(block)
            else:
                self.current.close()
                self.current = None
        return b"".join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def restore(bundle: Path, parts: list[Path], expected: dict) -> int:
    seen = set()
    with PartsReader(parts) as source, tarfile.open(fileobj=source, mode="r|gz") as archive:
        for member in archive:
            if member.name not in expected or not member.isfile():
                raise ValueError(f"Unexpected archive member: {member.name}")
            if member.name in seen:
                raise ValueError(f"Repeated archive member: {member.name}")
            seen.add(member.name)
            target = bundle / member.name
            if not target.resolve().is_relative_to(bundle.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            record = expected[member.name]
            if member.size != int(record["bytes"]):
                raise ValueError(f"Unexpected archive size: {member.name}")
            if target.is_file() and target.stat().st_size == member.size and sha256(target) == record["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".release-download-tmp")
            try:
                with archive.extractfile(member) as stream, temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, 4 * 1024**2)
                if sha256(temporary) != record["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {member.name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            if len(seen) % 1000 == 0:
                print(f"Restored {len(seen)}/{len(expected)} files", flush=True)
    if seen != expected.keys():
        raise ValueError(f"Archive is missing {len(expected.keys()-seen)} expected files")
    return len(seen)


def verify(bundle: Path, expected: dict) -> list[str]:
    errors = []
    for name, record in expected.items():
        path = bundle / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            errors.append(name)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify already restored data without downloading")
    args = parser.parse_args()
    data = Path(__file__).resolve().parent
    bundle = data.parent
    metadata = json.loads((data / "DATA_RELEASE.json").read_text())
    with (data / "DATA_MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
        expected = {r["path"]: r for r in csv.DictReader(stream)}
    if args.verify_only:
        errors = verify(bundle, expected)
        print(f"Verified {len(expected)-len(errors)}/{len(expected)} data files")
        if errors:
            print("Missing or changed:", *errors[:20], sep="\n")
            raise SystemExit(1)
        return
    cache = data / ".downloads"
    cache.mkdir(exist_ok=True)
    parts = []
    for item in metadata["archive_parts"]:
        path = cache / item["name"]
        if not path.exists() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            temporary = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {item['name']} ({item['bytes']/1024**2:.1f} MiB)", flush=True)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "DOVE-data-downloader"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 4 * 1024**2)
            if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Download verification failed: {item['name']}")
            temporary.replace(path)
        parts.append(path)
    restored = restore(bundle, parts, expected)
    print(f"Restored and verified {restored} data files. The main DOVE simulation checkpoint is not distributed.")


if __name__ == "__main__":
    main()
