"""Environment checks shared by the desktop dependency panel."""
from __future__ import annotations

import os
from pathlib import Path


RUNTIME_DEPENDENCIES = (
    ("NumPy 数值计算", "numpy"),
    ("OpenCV 摄像头与图像处理", "cv2"),
    ("MediaPipe 面部特征", "mediapipe"),
    ("pandas 数据表", "pandas"),
    ("openpyxl 工作簿", "openpyxl"),
    ("Pillow 图像处理", "PIL"),
    ("scikit-learn 校准模型", "sklearn"),
    ("SciPy 科学计算", "scipy"),
    ("screeninfo 显示器信息", "screeninfo"),
    ("xlrd 旧版 Excel 工作簿", "xlrd"),
)


def dependency_install_command(
    project_root: Path,
    prefix: str,
    executable: str,
    *,
    frozen: bool,
) -> tuple[tuple[str, ...] | None, str]:
    """Return the one supported dependency command, or its disable reason."""
    root = project_root.resolve()
    project_venv = root / ".venv"
    if frozen:
        return None, "便携版不能在运行目录中安装源码依赖；请解压完整便携包或使用源码启动器。"
    if os.path.normcase(str(Path(prefix).resolve())) != os.path.normcase(str(project_venv.resolve())):
        return None, f"仅在项目自己的 .venv 中安装依赖。当前环境：{prefix}"

    python = Path(executable).resolve()
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    requirements = root / "requirements.txt"
    if not python.is_file() or not requirements.is_file():
        return None, "项目 .venv 的 Python 或项目 requirements.txt 不可用。"
    return (str(python), "-m", "pip", "install", "-r", str(requirements)), "将只按项目 requirements.txt 安装到当前项目 .venv。"
