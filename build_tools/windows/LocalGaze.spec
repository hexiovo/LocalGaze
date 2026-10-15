"""Three Windows entry points sharing one complete, portable runtime folder."""
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).resolve().parents[1]
datas = [(str(root / "README.md"), ".")]
user_manual = root / "docs" / "LocalGaze产品介绍与使用手册.pdf"
if not user_manual.is_file():
    raise RuntimeError(f"Product and user manual is missing: {user_manual}")
datas += [(str(user_manual), "docs")]
datas += [(str(root / "assets" / "localgaze.ico"), "assets"),
          (str(root / "assets" / "localgaze.png"), "assets")]
datas += [(str(path), "assets/guide") for path in sorted((root / "assets" / "guide").glob("*.png"))]
if not any(Path(source).name == "01-workbench.png" for source, _ in datas):
    raise RuntimeError("Current guide screenshots are missing from assets/guide")
datas += collect_data_files("mediapipe")
for package in ("mediapipe", "numpy", "scikit-learn", "opencv-contrib-python", "pandas"):
    datas += copy_metadata(package)

hiddenimports = [
    "capture_persistence", "capture_export_recovery", "job_logging",
    "eyetrax.movement", "eyetrax.movement_runtime",
    "eyetracking_pipeline", "eyetracking", "misc", "Global_data",
    "eyetrax.gaze", "eyetrax.data_contract", "eyetrax.models.ridge",
    "eyetrax.models.elastic_net", "eyetrax.models.svr", "eyetrax.models.tiny_mlp",
    "eyetrax.calibration", "eyetrax.calibration.adaptive",
    "ROIDrawing.roi_drawer", "ROIDrawing.ClassType", "ROIDrawing.Global_data",
    "EyeEvaluate.Evaluate.analysis_pipeline", "EyeEvaluate.Evaluate.analysis_options",
    "EyeEvaluate.FileOperate.ROIextract",
    "pre.pipeline", "pre.fileio", "openpyxl", "screeninfo", "desktop_diagnostics",
]

a = Analysis(
    [str(root / "launch.py")], pathex=[str(root), str(root / "LocalGaze")],
    binaries=collect_dynamic_libs("mediapipe"), datas=datas,
    hiddenimports=hiddenimports,
    excludes=["IPython", "notebook", "pytest", "tensorflow", "torch", "pyvirtualcam"],
    noarchive=False,
)
# Conda base can be earlier on PATH than this interpreter's native libraries.
# Pin Tk's DLL pair to the interpreter whose Tcl/Tk scripts the hook collected.
for dll_name in ("tcl86t.dll", "tk86t.dll"):
    candidates = [base / directory / dll_name
                  for base in dict.fromkeys((Path(sys.prefix), Path(sys.base_prefix)))
                  for directory in (Path("Library") / "bin", Path("DLLs"))]
    dll_path = next((p for p in candidates if p.is_file()), None)
    if dll_path is None:
        raise RuntimeError(f"Cannot locate matching {dll_name} in {sys.prefix}")
    a.binaries = [(dest, str(dll_path), kind) if Path(dest).name.lower() == dll_name else (dest, source, kind)
                  for dest, source, kind in a.binaries]
pyz = PYZ(a.pure)
common = dict(exclude_binaries=True, debug=False, strip=False, upx=False,
              icon=str(root / "assets" / "localgaze.ico"),
              version=str(root / "build_tools" / "windows" / "version_info.txt"))
app = EXE(pyz, a.scripts, [], name="LocalGaze", console=False, **common)
worker = EXE(pyz, a.scripts, [], name="LocalGazeWorker", console=True, **common)
roi = EXE(pyz, a.scripts, [], name="ROIDrawing", console=False, **common)
coll = COLLECT(app, worker, roi, a.binaries, a.datas,
               strip=False, upx=False, name="LocalGaze")
