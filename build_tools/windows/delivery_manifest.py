"""Write package hashes outside the archive so the ZIP does not hash itself."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--attachments-root", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    release, archive, output = args.release, args.archive, args.output
    bundled_manifest = json.loads((release / "release-manifest.json").read_text(encoding="utf-8"))
    attachments = []
    if args.attachments_root:
        attachments = sorted(path for path in args.attachments_root.iterdir() if path.is_file())
    payload = {
        "release": release.name,
        "project_version": bundled_manifest.get("project_version"),
        "archive": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "release_manifest_sha256": digest(release / "release-manifest.json"),
        "executables_sha256": {
            name: digest(release / name)
            for name in ("LocalGaze.exe", "LocalGazeWorker.exe", "ROIDrawing.exe")
        },
        "version_file_sha256": digest(release / "version.md"),
        "delivery_manifest_is_external": True,
        "attachments": [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path)}
            for path in attachments
        ],
    }
    mode = "w" if args.replace else "x"
    with output.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    checksums = output.parent / "checksums.sha256"
    checksum_files = [archive, *attachments, output]
    with checksums.open(mode, encoding="utf-8") as stream:
        for path in checksum_files:
            stream.write(f"{digest(path)}  {path.name}\n")


if __name__ == "__main__":
    main()
