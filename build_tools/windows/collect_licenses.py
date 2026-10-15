"""Collect original notices for the Windows build and source distribution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import tkinter
import _tkinter
from urllib.parse import quote, urlparse


LICENSE_TOKENS = ("license", "licence", "copying", "notice")
LICENSE_SUFFIXES = {"", ".txt", ".md", ".rst", ".terms", ".external", ".html"}
PINNED_LICENSES = {
    "flatbuffers": {
        "version": "25.12.19",
        "file": "flatbuffers-25.12.19-LICENSE",
        "source": "https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE",
        "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
    "tcl": {
        "version": "8.6.12",
        "file": "tcl-8.6.12-license.terms",
        "source": "https://github.com/tcltk/tcl/blob/core-8-6-12/license.terms",
        "sha256": "c0a69a2bfd757361ec7e6143973b103c90409316b49e9c88db26ad6388e79f16",
    },
    "eyetrax": {
        "version": "0.3.1",
        "upstream_version": "1.0.0",
        "file": "eyetrax-1.0.0-LICENSE",
        "source": "https://github.com/ck-zhang/EyeTrax/blob/v1.0.0/LICENSE",
        "sha256": "dffadbb9e0e2b9024c2849f48470854bd46374e22c03db72406b6cbab975b3f1",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    if not normalized or normalized in {".", ".."}:
        raise RuntimeError(f"Unsafe component name: {value!r}")
    return normalized


def license_candidate(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in LICENSE_SUFFIXES and any(token in name for token in LICENSE_TOKENS)


def package_source(distribution: importlib.metadata.Distribution, name: str) -> str:
    preferred = ("source", "repository", "homepage", "home page")
    project_urls = distribution.metadata.get_all("Project-URL") or []
    choices: list[tuple[str, str]] = []
    for value in project_urls:
        label, separator, url = value.partition(",")
        if separator:
            choices.append((label.strip().lower(), url.strip()))
    for target in preferred:
        for label, url in choices:
            if label == target and urlparse(url).scheme in {"https", "http"}:
                return url
    home_page = distribution.metadata.get("Home-page")
    if home_page and urlparse(home_page).scheme in {"https", "http"}:
        return home_page
    normalized = quote(name.replace("_", "-"), safe="-")
    return f"https://pypi.org/project/{normalized}/{distribution.version}/"


def pinned_source_file(project: Path, entry: dict[str, str]) -> Path:
    path = project / "build_tools" / "windows" / "license_sources" / entry["file"]
    if not path.is_file():
        raise RuntimeError(f"Pinned upstream license file is missing: {entry['file']}")
    actual = sha256(path)
    if actual != entry["sha256"]:
        raise RuntimeError(
            f"Pinned upstream license changed: {entry['file']} has SHA-256 {actual}"
        )
    return path


def copy_license(source: Path, destination: Path, relative: str) -> str:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative or ":" in relative:
        raise RuntimeError(f"Unsafe license path from package metadata: {relative!r}")
    resolved = source.resolve(strict=True)
    destination_path = destination.joinpath(*parsed.parts)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, destination_path)
    return destination_path.relative_to(destination).as_posix()


def add_external_license(
    component: dict[str, object], source: Path, output: Path, name: str
) -> None:
    relative = f"licenses/upstream/{safe_component(name)}/{source.name}"
    copied = copy_license(source, output, relative)
    component["license_files"].append(copied)


def collect(project: Path, output: Path, notices_path: Path) -> dict[str, object]:
    project = project.resolve()
    output = output.resolve()
    notices_path = notices_path.resolve()
    if output == project or not output.is_relative_to(project):
        raise RuntimeError("The generated license directory must be inside the project")
    if output.exists():
        managed_manifest = output / "components.json"
        try:
            existing_manifest = json.loads(managed_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = None
        current_output = (
            isinstance(existing_manifest, dict)
            and existing_manifest.get("generated_by") == "build_tools/windows/collect_licenses.py"
            and existing_manifest.get("schema_version") == 1
        )
        previous_output = (
            isinstance(existing_manifest, dict)
            and existing_manifest.get("schema_version") == 1
            and (output / ".codex-managed-license-output").is_file()
        )
        if not (current_output or previous_output):
            raise RuntimeError(f"Refusing to replace unmanaged directory: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix=".license-collection-", dir=output.parent))
    try:
        components: list[dict[str, object]] = []
        distributions = sorted(
            importlib.metadata.distributions(),
            key=lambda item: (
                (item.metadata.get("Name") or "").casefold(),
                item.version,
            ),
        )
        if not distributions:
            raise RuntimeError("No installed Python distributions were found")
        seen: set[tuple[str, str]] = set()
        flatbuffers_added = False

        for distribution in distributions:
            name = distribution.metadata.get("Name")
            if not name:
                raise RuntimeError(f"Distribution {distribution!r} has no Name metadata")
            key = (name.casefold().replace("_", "-"), distribution.version)
            if key in seen:
                continue
            seen.add(key)

            files = sorted(
                {
                    PurePosixPath(str(item))
                    for item in (distribution.files or [])
                    if license_candidate(PurePosixPath(str(item)))
                },
                key=lambda item: item.as_posix().casefold(),
            )
            component: dict[str, object] = {
                "name": name,
                "version": distribution.version,
                "source": package_source(distribution, name),
                "license_files": [],
            }
            slug = safe_component(f"{name}-{distribution.version}")
            for relative in files:
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe package license path: {relative}")
                source = Path(distribution.locate_file(relative))
                if not source.is_file():
                    continue
                target_relative = PurePosixPath("licenses", "python", slug, *relative.parts)
                copied = copy_license(source, build_dir, target_relative.as_posix())
                component["license_files"].append(copied)

            if not component["license_files"]:
                if key[0] != "flatbuffers":
                    raise RuntimeError(
                        f"No original license text found for {name}=={distribution.version}"
                    )
                pinned = PINNED_LICENSES["flatbuffers"]
                if distribution.version != pinned["version"]:
                    raise RuntimeError(
                        f"flatbuffers=={distribution.version} has no installed license file; "
                        f"the verified license covers only {pinned['version']}"
                    )
                source = pinned_source_file(project, pinned)
                add_external_license(component, source, build_dir, "flatbuffers")
                component["source"] = pinned["source"]
                flatbuffers_added = True
            components.append(component)

        if not flatbuffers_added:
            flatbuffers = next(
                (item for item in components if str(item["name"]).casefold() == "flatbuffers"),
                None,
            )
            if flatbuffers is None:
                raise RuntimeError("flatbuffers is absent from the build environment")

        base = Path(sys.base_prefix).resolve()
        python_license = base / "LICENSE.txt"
        if not python_license.is_file():
            raise RuntimeError(f"The Python runtime license was not found: {python_license}")
        components.append({
            "name": "CPython",
            "version": sys.version.split()[0],
            "source": "https://www.python.org/downloads/",
            "license_files": [copy_license(
                python_license, build_dir, "licenses/runtime/CPython-LICENSE.txt"
            )],
        })

        tcl_version = str(tkinter.Tcl().eval("info patchlevel"))
        pinned_tcl = PINNED_LICENSES["tcl"]
        if tcl_version != pinned_tcl["version"]:
            raise RuntimeError(
                f"The Tcl runtime is {tcl_version}; no pinned license source is available "
                f"for that exact version (verified: {pinned_tcl['version']})"
            )
        tcl_source = pinned_source_file(project, pinned_tcl)
        tcl_component: dict[str, object] = {
            "name": "Tcl",
            "version": tcl_version,
            "source": pinned_tcl["source"],
            "license_files": [],
        }
        add_external_license(tcl_component, tcl_source, build_dir, "Tcl")
        components.append(tcl_component)

        tk_version = f"{_tkinter.TK_VERSION}.{tcl_version.rsplit('.', 1)[-1]}"
        tk_license = base / "tcl" / f"tk{_tkinter.TK_VERSION}" / "license.terms"
        if not tk_license.is_file():
            raise RuntimeError(f"The Tk runtime license was not found: {tk_license}")
        components.append({
            "name": "Tk",
            "version": tk_version,
            "source": "https://www.tcl.tk/software/tcltk/license.html",
            "license_files": [copy_license(
                tk_license, build_dir, "licenses/runtime/Tk-license.terms"
            )],
        })

        eyetrax_version_source = project / "LocalGaze" / "eyetrax" / "_version.py"
        version_source = eyetrax_version_source.read_text(encoding="utf-8")
        version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', version_source, re.MULTILINE)
        eyetrax = PINNED_LICENSES["eyetrax"]
        if version_match is None or version_match.group(1) != eyetrax["version"]:
            raise RuntimeError("The vendored EyeTrax version does not match its verified source mapping")
        eyetrax_license = pinned_source_file(project, eyetrax)
        eyetrax_component: dict[str, object] = {
            "name": "EyeTrax (vendored)",
            "version": eyetrax["version"],
            "source": eyetrax["source"],
            "license_files": [],
        }
        add_external_license(eyetrax_component, eyetrax_license, build_dir, "EyeTrax")
        eyetrax_component["upstream_tag"] = eyetrax["upstream_version"]
        components.append(eyetrax_component)

        manifest = {
            "schema_version": 1,
            "generated_by": "build_tools/windows/collect_licenses.py",
            "components": components,
        }
        (build_dir / "components.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        notices = render_notices(components)
        notices_temp = notices_path.with_name(notices_path.name + ".tmp")
        notices_temp.parent.mkdir(parents=True, exist_ok=True)
        notices_temp.write_text(notices, encoding="utf-8")

        if output.exists():
            shutil.rmtree(output)
        os.replace(build_dir, output)
        os.replace(notices_temp, notices_path)
        return manifest
    except Exception:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        raise


def render_notices(components: list[dict[str, object]]) -> str:
    rows = [
        "# Third-party notices",
        "",
        "LocalGaze's own source code has no project-level license. The notices below apply only to the named third-party components. The original license texts are included in `third_party/licenses/`.",
        "",
        "| Component | Version | Source | Original license text |",
        "| --- | --- | --- | --- |",
    ]
    for component in components:
        source = str(component["source"])
        links = "<br>".join(
            f"[`{Path(str(path)).name}`](third_party/{path})"
            for path in component["license_files"]
        )
        rows.append(
            f"| {component['name']} | {component['version']} | [source]({source}) | {links} |"
        )
    rows.append("")
    rows.append(
        "This index is generated from the build interpreter's installed distribution metadata and the pinned upstream license sources under `build_tools/windows/license_sources/`. It records build-environment packages as well as runtime components so the accompanying license texts remain available for review."
    )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = collect(
        args.project,
        args.output,
        args.project / "THIRD_PARTY_NOTICES.md",
    )
    print(json.dumps({"components": len(manifest["components"]), "status": "passed"}))


if __name__ == "__main__":
    main()
