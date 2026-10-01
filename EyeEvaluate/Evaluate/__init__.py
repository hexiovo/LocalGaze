from .ROIdefine import (
    all_hits,
    classify_roi_point,
    identify_roi,
    process_roi_lists,
)

__all__ = [
    "all_hits",
    "classify_roi_point",
    "identify_roi",
    "process_roi_lists",
    "AnalysisConfig",
    "AnalysisResult",
    "analyze_file",
    "analyze_samples",
    "export_analysis_result",
]


def __getattr__(name):
    if name in {"AnalysisConfig", "AnalysisResult", "analyze_samples"}:
        from . import metrics

        value = getattr(metrics, name)
    elif name == "analyze_file":
        from .analysis_pipeline import analyze_file

        value = analyze_file
    elif name == "export_analysis_result":
        from .analysis_export import export_analysis_result

        value = export_analysis_result
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
