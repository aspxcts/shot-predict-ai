"""Hue-agnostic tennis ball candidates and tracklets from a steady camera.

Broadcast balls are small, motion-blurred and washed out (on the US Open footage a pale-green streak
about 12x6 px), so color is not used. A pixel belongs to the ball at frame t when it differs from BOTH
the previous and the next frame (three-frame differencing: the ball's old and new positions each
differ from only one neighbour). Candidates are linked into tracklets with a velocity-gated tracker.
Missing detections stay missing; nothing is interpolated.
"""

import cv2
import numpy as np

DIFF_THRESHOLD = 16
MIN_AREA = 3
MAX_AREA = 260
MAX_SIDE = 48
CROWD_WINDOW = 60
CROWD_CELL = 4
CROWD_FACTOR = 8.0
CROWD_BASE = 200.0
GATE_PIXELS_PER_SECOND = 3600.0
MAX_GAP_SECONDS = 0.12  # 3 frames at 25 fps; longer gaps (5 frames) measurably chained noise into false flights
MIN_TRACKLET_POINTS = 5
MIN_TRACKLET_DISPLACEMENT = 40.0
FLAT_TRACK_Y_RANGE = 4.0
FLAT_TRACK_MIN_WIDTH = 60.0
MAX_TRACK_ROUGHNESS = 6.0
MAX_STEP_FRACTION = 0.5  # of the prediction slack that may be spent on the sudden velocity change at a kink


def court_roi(corners: np.ndarray, size: tuple[int, int], margin_x=0.4, margin_near=0.18, margin_far=0.5) -> np.ndarray:
    """255 inside the court plus margins (ball height shifts it toward the far side of the image).

    Court lines and the net are deliberately NOT carved out: that also removes the ball around the baselines and
    net, which cut contact recall within 200 ms from 79% to 57% on Turin footage.
    """
    to_image = cv2.getPerspectiveTransform(np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32), np.asarray(corners, np.float32))
    court = np.array([[[-margin_x, 1 + margin_near], [1 + margin_x, 1 + margin_near],
                       [1 + margin_x, -margin_far], [-margin_x, -margin_far]]], np.float32)
    polygon = cv2.perspectiveTransform(court, to_image).reshape(-1, 2).round().astype(np.int32)
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def ball_candidates(prev, cur, nxt, roi: np.ndarray, blocked: list | None = None, threshold: int = DIFF_THRESHOLD):
    """Small moving blobs at frame `cur`: list of (x, y, area, strength). Accepts BGR or gray frames."""
    gray = lambda image: image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return ball_candidates_gray(gray(prev), gray(cur), gray(nxt), roi, blocked, threshold)


def ball_candidates_gray(prev, cur, nxt, roi: np.ndarray, blocked: list | None = None, threshold: int = DIFF_THRESHOLD):
    x0, y0, width, height = cv2.boundingRect(roi)
    if width == 0 or height == 0:
        return []
    window = (slice(y0, y0 + height), slice(x0, x0 + width))
    cur_w = cur[window]
    first = cv2.absdiff(cur_w, prev[window])
    second = cv2.absdiff(cur_w, nxt[window])
    both = cv2.min(first, second)
    mask = ((both > threshold).astype(np.uint8)) * 255
    mask[roi[window] == 0] = 0
    for box in blocked or ():
        if box is not None:
            bx1, by1, bx2, by2 = (int(round(v)) for v in box[:4])
            cv2.rectangle(mask, (bx1 - x0, by1 - y0), (bx2 - x0, by2 - y0), 0, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return []
    # A ball is nearly alone in its neighbourhood; the edges of a large moving object are not.
    moving = (cv2.max(first, second) > threshold).astype(np.uint8)
    cell = CROWD_CELL
    coarse = cv2.resize(moving, (max(1, width // cell), max(1, height // cell)), interpolation=cv2.INTER_AREA).astype(np.float32)
    crowding = cv2.boxFilter(coarse, -1, (CROWD_WINDOW // cell, CROWD_WINDOW // cell), normalize=False) * cell * cell
    # Mean difference per component in one pass (a per-component window scan is what made noisy frames slow).
    totals = np.bincount(labels.ravel(), weights=both.ravel().astype(np.float64), minlength=count)
    found = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (MIN_AREA <= area <= MAX_AREA) or max(w, h) > MAX_SIDE:
            continue
        cx, cy = centres[index]
        row = min(max(int(cy) // cell, 0), crowding.shape[0] - 1)
        column = min(max(int(cx) // cell, 0), crowding.shape[1] - 1)
        if crowding[row, column] > CROWD_FACTOR * area + CROWD_BASE:
            continue
        found.append((float(cx + x0), float(cy + y0), int(area), float(totals[index] / area)))
    return found


def persistent_noise_cells(candidates: dict[int, list], size: tuple[int, int], cell: int = 8, fraction: float = 0.06, minimum: int = 20):
    """Grid cells that hold candidates in many frames: signage, line judges and net logos, never a flying ball."""
    columns, rows = size[0] // cell + 1, size[1] // cell + 1
    counts = np.zeros((rows, columns), np.float32)
    for items in candidates.values():
        seen = {(min(int(y) // cell, rows - 1), min(int(x) // cell, columns - 1)) for x, y, *_ in items}
        for row, column in seen:
            counts[row, column] += 1
    # Neighbouring cells share flicker that straddles a cell boundary.
    smoothed = cv2.boxFilter(counts, -1, (3, 3), normalize=False)
    threshold = max(minimum, fraction * max(1, len(candidates)))
    return smoothed >= threshold, cell


def drop_persistent_noise(candidates: dict[int, list], size: tuple[int, int], **kwargs) -> dict[int, list]:
    noisy, cell = persistent_noise_cells(candidates, size, **kwargs)
    rows, columns = noisy.shape
    return {
        frame: [c for c in items if not noisy[min(int(c[1]) // cell, rows - 1), min(int(c[0]) // cell, columns - 1)]]
        for frame, items in candidates.items()
    }


def expand_box(box, fraction: float = 0.18, minimum: int = 8):
    """Person box grown to cover racket and limbs that the detector clips."""
    if box is None:
        return None
    x1, y1, x2, y2 = box[:4]
    margin = max(minimum, fraction * (y2 - y1))
    return [x1 - margin, y1 - margin, x2 + margin, y2 + margin]


class _Track:
    def __init__(self, frame, point):
        self.points = [(frame, point[0], point[1])]

    @property
    def last_frame(self):
        return self.points[-1][0]

    def predict(self, frame):
        f1, x1, y1 = self.points[-1]
        if len(self.points) < 2:
            return np.array([x1, y1]), 0.0
        f0, x0, y0 = self.points[-2]
        velocity = np.array([x1 - x0, y1 - y0]) / (f1 - f0)
        return np.array([x1, y1]) + velocity * (frame - f1), float(np.linalg.norm(velocity))


def _roughness(points: np.ndarray) -> float:
    """Median distance of each interior point from the time-weighted line between its neighbours (pixels)."""
    deviations = []
    for i in range(1, len(points) - 1):
        (f0, x0, y0), (f1, x1, y1), (f2, x2, y2) = points[i - 1], points[i], points[i + 1]
        weight = (f1 - f0) / (f2 - f0)
        deviations.append(np.hypot(x1 - (x0 + weight * (x2 - x0)), y1 - (y0 + weight * (y2 - y0))))
    return float(np.median(deviations)) if deviations else 0.0


def link_tracklets(candidates: dict[int, list], fps: float) -> list[dict]:
    """Greedy velocity-gated tracker over frames. `candidates` maps frame -> [(x, y, area, strength)]."""
    if not candidates:
        return []
    gate = GATE_PIXELS_PER_SECOND / fps
    max_missed = max(2, round(MAX_GAP_SECONDS * fps))
    active: list[_Track] = []
    finished: list[_Track] = []
    for frame in range(min(candidates), max(candidates) + 1):
        for track in [t for t in active if frame - t.last_frame > max_missed]:
            active.remove(track)
            finished.append(track)
        points = candidates.get(frame, [])
        pairs = []
        for ti, track in enumerate(active):
            gap = frame - track.last_frame
            predicted, speed = track.predict(frame)
            # A one-point track has no velocity yet, so only the speed limit can constrain its first link.
            slack = gate * gap if len(track.points) < 2 else 14.0 + MAX_STEP_FRACTION * 1.2 * speed * gap
            for ci, point in enumerate(points):
                xy = np.array(point[:2])
                last = np.array(track.points[-1][1:])
                if np.linalg.norm(xy - last) <= gate * gap and np.linalg.norm(xy - predicted) <= slack:
                    pairs.append((float(np.linalg.norm(xy - predicted)), ti, ci))
        used_tracks, used_points = set(), set()
        for _, ti, ci in sorted(pairs):
            if ti not in used_tracks and ci not in used_points:
                used_tracks.add(ti)
                used_points.add(ci)
                active[ti].points.append((frame, points[ci][0], points[ci][1]))
        for ci, point in enumerate(points):
            if ci not in used_points:
                active.append(_Track(frame, point))
    finished.extend(active)
    tracklets = []
    for track in finished:
        pts = np.array(track.points, float)
        if len(pts) < MIN_TRACKLET_POINTS:
            continue
        if np.linalg.norm(pts[-1, 1:] - pts[0, 1:]) < MIN_TRACKLET_DISPLACEMENT:
            continue
        # Court-line shimmer links into long, perfectly flat "flights"; a ball always arcs.
        if np.ptp(pts[:, 2]) < FLAT_TRACK_Y_RANGE and np.ptp(pts[:, 1]) > FLAT_TRACK_MIN_WIDTH:
            continue
        # Chance coincidences chain into jagged paths. A flying ball is locally smooth at almost every point; a
        # bounce or racket contact is one outlier, so the median of local deviations ignores it. Curvature per frame
        # grows with the square of the frame time, so the limit is scaled from its 60 fps value.
        if _roughness(pts) > min(MAX_TRACK_ROUGHNESS * (60.0 / fps) ** 2, 40.0):
            continue
        tracklets.append({"frames": pts[:, 0].astype(int).tolist(), "xy": pts[:, 1:].round(2).tolist()})
    tracklets.sort(key=lambda t: t["frames"][0])
    return tracklets


def scan_candidates(path, start: int, end: int, roi_for_frame, boxes_for_frame, progress=None) -> dict[int, list]:
    """Candidates for frames [start, end) read sequentially (needs frames start-1 .. end)."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start - 1))
        frames = []
        for _ in range(3):
            ok, frame = cap.read()
            if not ok:
                return {}
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = {}
        index = max(0, start - 1) + 1
        while index < end:
            roi = roi_for_frame(index)
            if roi is not None:
                found[index] = ball_candidates_gray(frames[0], frames[1], frames[2], roi, boxes_for_frame(index))
            if progress and (index - start) % 600 == 0:
                progress(index)
            ok, frame = cap.read()
            if not ok:
                break
            frames = [frames[1], frames[2], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]
            index += 1
        return found
    finally:
        cap.release()
