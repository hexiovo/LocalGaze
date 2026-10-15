"""Verify the portable payload and keep source-integrity details outside it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import pefile


PUBLIC_KEYS = {
    "schema_version", "release", "project_version", "platform", "architecture",
    "python_version", "analysis_method_version", "eyetrax_version",
    "default_feature_formula", "version_file_sha256", "dependencies", "files",
    "verification", "validation_limits",
}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise AssertionError(f"Not a public relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AssertionError(f"Not a public relative path: {value!r}")
    return path


def ico_frames(path):
    data = Path(path).read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind) == (0, 1)
    result = {}
    for index in range(count):
        width, height, _, _, _, _, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * index)
        result[(width or 256, height or 256)] = hashlib.sha256(data[offset:offset + length]).hexdigest()
    return result


def entry_info(path, expected_icons, subsystem):
    with pefile.PE(str(path)) as pe:
        assert pe.FILE_HEADER.Machine == 0x8664, "Not an x64 executable"
        assert pe.OPTIONAL_HEADER.Subsystem == subsystem, "Wrong Windows subsystem"
        resources = {entry.id: entry for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries}
        image_data = {}
        for entry in resources[3].directory.entries:
            data = entry.directory.entries[0].data.struct
            image_data[entry.id] = pe.get_data(data.OffsetToData, data.Size)
        groups = []
        for entry in resources[14].directory.entries:
            resource = entry.directory.entries[0].data.struct
            data = pe.get_data(resource.OffsetToData, resource.Size)
            reserved, kind, count = struct.unpack_from("<HHH", data)
            assert (reserved, kind) == (0, 1)
            frames = {}
            for index in range(count):
                width, height, _, _, _, _, _, identifier = struct.unpack_from("<BBBBHHIH", data, 6 + 14 * index)
                frames[(width or 256, height or 256)] = hashlib.sha256(image_data[identifier]).hexdigest()
            groups.append(frames)
        assert expected_icons in groups, "Embedded frames do not match the selected artwork"
        version = pe.VS_FIXEDFILEINFO[0]
        actual_version = [version.FileVersionMS >> 16, version.FileVersionMS & 0xffff,
                          version.FileVersionLS >> 16, version.FileVersionLS & 0xffff]
        assert actual_version == [2026, 10, 15, 0], actual_version
        return {"sha256": digest(path), "bytes": path.stat().st_size,
                "x64": True, "subsystem": subsystem, "file_version": actual_version,
                "exact_embedded_icon_sizes": sorted(expected_icons)}


def verify_public_manifest(bundle):
    manifest_path = bundle / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == PUBLIC_KEYS, f"Unexpected public manifest fields: {sorted(set(manifest) ^ PUBLIC_KEYS)}"
    assert manifest["schema_version"] == 1
    assert manifest["release"] == bundle.name
    assert manifest["platform"] == "windows"
    assert manifest["architecture"] in {"x86_64", "amd64"}
    assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+", manifest["project_version"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["python_version"])
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert not re.search(r"(?i)\b[a-z]:[\\/]", serialized), "Public manifest contains a drive-qualified path"
    assert "\\" not in serialized, "Public manifest contains a backslash or escaped machine path"
    assert manifest["verification"].get("status") == "passed"
    assert manifest["verification"].get("checks_passed") == manifest["verification"].get("checks_total")
    assert manifest["verification"].get("checks_total", 0) > 0

    listed = {}
    for item in manifest["files"]:
        relative = safe_relative(item.get("path"))
        key = relative.as_posix()
        assert key not in listed, f"Duplicate public manifest entry: {key}"
        if key == "release-manifest.json":
            raise AssertionError("release-manifest.json cannot hash itself")
        target = bundle.joinpath(*relative.parts)
        assert target.is_file(), f"Missing bundle file: {key}"
        assert target.stat().st_size == item.get("size_bytes"), f"Size mismatch: {key}"
        assert digest(target) == item.get("sha256"), f"Hash mismatch: {key}"
        listed[key] = item
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "release-manifest.json"
    }
    assert actual == set(listed), f"Bundle inventory mismatch: missing={sorted(actual-set(listed))[:5]}, extra={sorted(set(listed)-actual)[:5]}"
    return manifest, len(listed)


def verify_source_inventory(project, source_inventory_path):
    source_manifest = json.loads(Path(source_inventory_path).read_text(encoding="utf-8"))
    assert source_manifest.get("schema_version") == 1
    sources = source_manifest.get("files_sha256")
    assert isinstance(sources, dict) and sources
    for relative, expected in sources.items():
        parts = safe_relative(relative)
        target = project.joinpath(*parts.parts)
        assert target.is_file(), f"Missing source file: {relative}"
        assert digest(target) == expected, f"Source drift: {relative}"
    return len(sources)


def main():
    if len(sys.argv) != 6:
        raise SystemExit("usage: verify_app_delivery.py PROJECT BUNDLE LAUNCHER REPORT SOURCE_INVENTORY")
    project, bundle, launcher, report, source_inventory_path = map(Path, sys.argv[1:])
    project, bundle, launcher = project.resolve(), bundle.resolve(), launcher.resolve()
    manifest, verified_bundle_files = verify_public_manifest(bundle)
    verified_sources = verify_source_inventory(project, source_inventory_path)
    expected_icons = ico_frames(project / "assets" / "localgaze.ico")
    entries = {str(launcher): entry_info(launcher, expected_icons, 2)}
    for name, subsystem in (("LocalGaze.exe", 2), ("LocalGazeWorker.exe", 3), ("ROIDrawing.exe", 2)):
        path = bundle / name
        entries[str(path)] = entry_info(path, expected_icons, subsystem)
    result = {
        "status": "passed",
        "bundle": str(bundle),
        "entry": str(launcher),
        "verified_bundle_files": verified_bundle_files,
        "verified_sources": verified_sources,
        "bundle_manifest_sha256": digest(bundle / "release-manifest.json"),
        "project_version": manifest["project_version"],
        "release_archive_candidate": manifest["release"],
        "license_files": sum(item["path"].startswith("third_party/") for item in manifest["files"]),
        "entries": entries,
        "limits": ["no real camera", "no clean-machine test",
                   "physical multi-monitor DPI behavior has not been verified"],
    }
    with Path(report).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "files": verified_bundle_files,
                      "sources": verified_sources, "entry": str(launcher)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
