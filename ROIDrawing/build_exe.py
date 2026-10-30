"""Build a single-file Windows application with embedded dependencies/notices."""
from pathlib import Path
import importlib.metadata
import json
import shutil
import sys
import PyInstaller.__main__
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
NOTICES = BASE / 'build' / 'licenses'
NOTICES.mkdir(parents=True, exist_ok=True)
versions = {}
for name in ('pandas', 'numpy', 'openpyxl', 'et_xmlfile', 'python-dateutil', 'pytz', 'six', 'tzdata', 'Pillow', 'pyinstaller'):
    dist = importlib.metadata.distribution(name)
    versions[name] = dist.version
    for file in dist.files or []:
        if any(word in str(file).lower() for word in ('license', 'copying', 'notice')):
            source = Path(dist.locate_file(file))
            if source.is_file():
                target = NOTICES / name / str(file).replace('../', '')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
for source in [Path(sys.base_prefix) / 'LICENSE.txt', *Path(sys.base_prefix).glob('tcl/*/license.terms')]:
    if source.exists():
        shutil.copy2(source, NOTICES / (source.parent.name + '-' + source.name))
(NOTICES / 'versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
image = Image.new('RGBA', (256, 256), '#FCF8F2')
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((24, 24, 232, 232), 32, fill='#20C46A', outline='#27324A', width=12)
draw.rectangle((62, 74, 194, 182), outline='#27324A', width=12)
for x, y in ((62, 74), (194, 74), (62, 182), (194, 182)):
    draw.rectangle((x-12, y-12, x+12, y+12), fill='#FCF8F2', outline='#27324A', width=5)
icon = BASE / 'roi.ico'
image.save(icon, sizes=[(n,n) for n in (16,20,24,30,32,36,40,44,48,64,96,128,256)])
PyInstaller.__main__.run([
    str(BASE / 'main.py'), '--name=ROIDrawing', '--onefile', '--windowed', '--noconfirm',
    '--distpath=' + str(BASE / 'dist'), '--workpath=' + str(BASE / 'build' / 'pyinstaller'),
    '--specpath=' + str(BASE), '--paths=' + str(BASE.parent),
    '--icon=' + str(icon), '--add-data=' + str(icon) + ';.',
    '--add-data=' + str(NOTICES) + ';licenses',
    '--hidden-import=openpyxl', '--hidden-import=smoke_test',
    '--exclude-module=EyeEvaluate.Popup', '--exclude-module=Popup',
    '--exclude-module=matplotlib', '--exclude-module=scipy', '--exclude-module=IPython',
    '--exclude-module=pytest', '--exclude-module=cv2', '--exclude-module=mediapipe',
])
exe = BASE / 'dist' / 'ROIDrawing.exe'
assert exe.is_file(), 'Build did not produce an EXE'
print(f'Built {exe} ({exe.stat().st_size:,} bytes)')
