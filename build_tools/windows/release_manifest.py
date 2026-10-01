"""Record exact bundled/source identities and the bounded validation result."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


if __name__ == '__main__':
    replace_existing = len(sys.argv) == 5 and sys.argv[4] == '--replace-manifest'
    if len(sys.argv) not in (4, 5):
        raise SystemExit('usage: release_manifest.py PROJECT RELEASE CHECK_REPORT [--replace-manifest]')
    project, release, report = map(Path, sys.argv[1:4])
    source = {}
    for path in project.rglob('*.py'):
        relative = path.relative_to(project)
        if relative.parts[0] == 'release' or '__pycache__' in relative.parts:
            continue
        source[relative.as_posix()] = digest(path)
    source['LocalGaze使用指导语及流程.txt'] = digest(project / 'LocalGaze使用指导语及流程.txt')
    for name in ('README.md', 'version.md', 'build_tools/windows/LocalGaze.spec', 'build_tools/windows/build.ps1',
                 'build_tools/windows/release_manifest.py', 'build_tools/windows/delivery_manifest.py',
                 'build_tools/windows/使用说明.txt'):
        source[name] = digest(project / name)
    files = {p.relative_to(release).as_posix(): digest(p) for p in sorted(release.rglob('*'))
             if p.is_file() and p.name != 'release-manifest.json'}
    check_result = json.loads(report.read_text(encoding='utf-8'))
    algorithm_check = next((item for item in check_result.get('checks', [])
                            if item.get('name') == 'bundled_algorithm_modules_and_version_metadata'), {})
    algorithm_details = algorithm_check.get('details', {})
    manifest = {
        'release': release.name, 'project_version': algorithm_details.get('eyetrax_version'), 'python': sys.version,
        'default_feature_formula': algorithm_details.get('default_feature_formula'),
        'version_file_sha256': source['version.md'],
        'dependencies': {name: importlib.metadata.version(name) for name in (
            'pyinstaller', 'mediapipe', 'opencv-contrib-python', 'numpy', 'pandas',
            'scipy', 'scikit-learn', 'openpyxl', 'pillow', 'screeninfo')},
        'sources_sha256': source, 'files_sha256': files,
        'installation_check': check_result,
        'validation_limits': ['no real-camera test', 'no independent gaze-accuracy measurement',
                              'no clean-machine test', 'no physical-DPI acceptance test'],
    }
    output = release / 'release-manifest.json'
    mode = 'w' if replace_existing else 'x'
    with output.open(mode, encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
