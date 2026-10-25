"""AutoCalibrator on rendered pitch views: registers from scratch, follows a pan, refuses blank grass."""

import cv2
import numpy as np

from src.serve.pipeline.auto_calibrator import AutoCalibrator, Pitch

PITCH = Pitch()
W, H = 640, 360
R180 = np.array([[-1, 0, PITCH.length], [0, -1, PITCH.width], [0, 0, 1.0]])
# Image -> pitch homography of an end-on broadcast view (data/sample.mp4 frame 0), scaled to 640x360.
VIEW = np.array([
    [0.001721552068997914, -0.09647301285092479, 58.72622705619442],
    [0.07600909292721926, 0.054312278195826076, -17.193953474327834],
    [3.2092442250735817e-06, 0.0015527216894611763, 1.0],
]) @ np.diag([2.0, 2.0, 1.0])


def _pan(px: float) -> np.ndarray:
    """Image -> pitch homography after the picture slid ``px`` pixels to the right."""
    return VIEW @ np.array([[1, 0, -px], [0, 1, 0], [0, 0, 1.0]])


def _render(view: np.ndarray) -> np.ndarray:
    """Grass up to 5 m around the pitch with 2 px white markings, stands beyond, seen through
    the image -> pitch homography ``view``."""
    ys, xs = np.mgrid[0:H, 0:W]
    p = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3) @ view.T
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    grass = p[:, 2] * p[-1, 2] > 0  # below the horizon, like the bottom-right pixel
    grass &= (x > -5) & (x < PITCH.length + 5) & (y > -5) & (y < PITCH.width + 5)
    image = np.where(grass.reshape(H, W, 1), np.uint8([40, 140, 50]), np.uint8([90, 80, 70]))
    G = np.linalg.inv(view)
    G /= G[2, 2]
    for poly in PITCH.polylines(0.25):
        p = np.c_[poly, np.ones(len(poly))] @ G.T
        front = p[:, 2] > 0  # every pitch point is in front of this camera
        xy = np.round(p[:, :2] / p[:, 2:3]).astype(np.int32)
        for a, b, ok in zip(xy[:-1], xy[1:], front[:-1] & front[1:]):
            if ok:
                cv2.line(image, tuple(a), tuple(b), (235, 235, 235), 2, cv2.LINE_AA)
    return image


def _error_m(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean pitch distance over visible pitch pixels, better of the two 180-degree labellings."""
    xs, ys = np.meshgrid(np.linspace(0, W - 1, 17), np.linspace(0, H - 1, 9))
    px = np.c_[xs.ravel(), ys.ravel()].astype(np.float32).reshape(-1, 1, 2)
    true_xy = cv2.perspectiveTransform(px, truth).reshape(-1, 2)
    on = (true_xy[:, 0] >= 0) & (true_xy[:, 0] <= PITCH.length)
    on &= (true_xy[:, 1] >= 0) & (true_xy[:, 1] <= PITCH.width)
    return min(
        float(np.linalg.norm(cv2.perspectiveTransform(px, F @ estimate).reshape(-1, 2) - true_xy, axis=1)[on].mean())
        for F in (np.eye(3), R180)
    )


def test_registers_a_view_from_its_markings() -> None:
    found = AutoCalibrator().update(_render(VIEW), 0)

    assert found is not None
    assert _error_m(found, VIEW) < 0.3


def test_follows_a_panning_camera() -> None:
    calib = AutoCalibrator()
    for n in range(11):
        found = calib.update(_render(_pan(3.0 * n)), n)

    assert found is not None
    assert _error_m(found, _pan(30.0)) < 0.3


def test_grass_without_markings_gets_no_homography() -> None:
    grass = np.full((H, W, 3), (40, 140, 50), np.uint8)

    assert AutoCalibrator().update(grass, 0) is None


def test_failed_searches_get_rarer() -> None:
    """Footage without a readable pitch must not pay for a search every few frames."""
    calib = AutoCalibrator(retry_frames=50)
    searched: list[int] = []
    calib._search = lambda image: searched.append(n) or None
    blank = np.zeros((H, W, 3), np.uint8)
    for n in range(2000):
        calib.update(blank, n)

    assert searched[:4] == [0, 50, 150, 350]
    assert len(searched) <= 6
