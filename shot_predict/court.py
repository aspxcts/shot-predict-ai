from dataclasses import dataclass

import cv2
import numpy as np

ZONES = (
    "ad_deep", "middle_deep", "deuce_deep",
    "ad_short", "middle_short", "deuce_short",
)
CORNER_NAMES = ("Near-left", "Near-right", "Far-right", "Far-left")
COURT_CORNERS = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)


def zone_for(x: float, y: float, hitter: str) -> str:
    """Receiver-relative landing zone for normalized court coordinates; independent of the camera."""
    if hitter not in ("near", "far"):
        raise ValueError("Hitter must be near or far.")
    if not np.isfinite([x, y]).all() or not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Bounce is outside the singles court; skip out shots.")
    if (hitter == "near" and y >= 0.5) or (hitter == "far" and y <= 0.5):
        raise ValueError("Bounce must be on the opposite side of the net.")
    # Ad/deuce are defined from the receiver's perspective facing the net.
    receiver_x = 1 - x if hitter == "near" else x
    lateral = "ad" if receiver_x < 1 / 3 else "deuce" if receiver_x >= 2 / 3 else "middle"
    depth = "deep" if (y < 0.25 if hitter == "near" else y > 0.75) else "short"
    return f"{lateral}_{depth}"


@dataclass
class Court:
    corners: list[list[float]]

    def __post_init__(self):
        points = np.asarray(self.corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Court needs four finite image points.")
        contour = points.reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 25:
            raise ValueError("Corners must form a convex court in the specified order.")
        self.to_court = cv2.getPerspectiveTransform(points, COURT_CORNERS)
        self.to_image = cv2.getPerspectiveTransform(COURT_CORNERS, points)

    def normalized(self, x: float, y: float) -> tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.to_court)[0, 0]
        return float(p[0]), float(p[1])

    def pixel(self, x: float, y: float) -> tuple[int, int]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.to_image)[0, 0]
        return round(float(p[0])), round(float(p[1]))

    def zone(self, x: float, y: float, hitter: str) -> str:
        return zone_for(x, y, hitter)

    def target(self, zone: str, hitter: str) -> tuple[int, int]:
        if zone not in ZONES or hitter not in ("near", "far"):
            raise ValueError("Invalid zone or hitter.")
        lateral, depth = zone.split("_")
        receiver_x = {"ad": 1 / 6, "middle": 0.5, "deuce": 5 / 6}[lateral]
        x = 1 - receiver_x if hitter == "near" else receiver_x
        y = (0.125 if depth == "deep" else 0.375) if hitter == "near" else (
            0.875 if depth == "deep" else 0.625
        )
        return self.pixel(x, y)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        for x in (0, 1 / 3, 2 / 3, 1):
            cv2.line(result, self.pixel(x, 0), self.pixel(x, 1), (70, 210, 70), 2)
        for y in (0, 0.25, 0.5, 0.75, 1):
            cv2.line(result, self.pixel(0, y), self.pixel(1, y), (70, 210, 70), 2)
        return result
