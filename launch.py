"""Source and frozen entry point for the workbench and its owned workers."""
from pathlib import Path
import sys


def main() -> int:
    # A frozen worker uses a console bootloader so its result pipe stays usable.
    # Dispatch before importing Tk; task children must never start another shell.
    if "--worker" in sys.argv[1:]:
        from desktop_jobs import main as worker_main
        return worker_main()
    if len(sys.argv) == 3 and sys.argv[1] == "--check-installation":
        from desktop_diagnostics import check_installation
        return check_installation(Path(sys.argv[2]))
    if getattr(sys, "frozen", False) and Path(sys.executable).stem.lower() == "roidrawing":
        from desktop_app import _enable_dpi_awareness
        from desktop_jobs import _roi_job
        _enable_dpi_awareness()
        _roi_job({"output_root": str(Path(sys.executable).resolve().parent)})
        return 0
    from desktop_app import main as desktop_main
    return desktop_main()


if __name__ == "__main__":
    raise SystemExit(main())
