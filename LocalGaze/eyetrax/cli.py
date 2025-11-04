import argparse


def parse_common_args():

    parser = argparse.ArgumentParser(description="Common Gaze Estimation Arguments")

    parser.add_argument(
        "--filter",
        choices=["kalman", "kde", "none"],
        default="none",
        help="Select the filter to apply to gaze estimation, options are 'kalman', 'kde', or 'none'",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for video capture, default is 0 (first camera)",
    )
    parser.add_argument(
        "--calibration",
        choices=["9p", "5p", "lissajous"],
        default="9p",
        help="Calibration method for gaze estimation, options are '9p', '5p', or 'lissajous'",
    )
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to a custom background image (optional)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence level for KDE smoothing, range 0 to 1",
    )
    parser.add_argument(
        "--model",
        default="ridge",
        help="The machine learning model to use for gaze estimation, default is 'ridge'",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to a previously-trained gaze model",
    )

    return parser.parse_args()
