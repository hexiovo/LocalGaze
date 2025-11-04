from __future__ import annotations

from contextlib import contextmanager

import cv2


@contextmanager
def fullscreen(name: str):
    """
    Open a window in full-screen mode
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        yield
    finally:
        cv2.destroyWindow(name)


@contextmanager
def camera(index: int = 0):
    """
    Context manager returning an opened VideoCapture
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")
    try:
        yield cap
    finally:
        cap.release()


def iter_frames(cap: cv2.VideoCapture):
    """
    Infinite generator yielding successive frames
    """
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        yield frame
