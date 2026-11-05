"""Camera layer: recover court corners on moving broadcast cameras and reject non-court views.

A tower camera pans and zooms from a fixed position, so any two of its frames are related by a
homography, including for background that is not on the court plane. A user-calibrated reference
frame can therefore be matched against later frames with local features; the reference's court
corners are carried across with that homography. A line check on the warped court then accepts
or rejects the result, which doubles as the court-view classifier (close-ups and crowd shots fail).
"""

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..video import read_frame

MATCH_WIDTH = 960
RATIO = 0.75
RANSAC_PIXELS = 4.0
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.25
MIN_LINE_SCORE = 0.55
MIN_LINE_POINTS = 40
SERVICE_FRACTIONS = (18 / 78, 60 / 78)  # service lines, as depth fractions of the singles court


def graphics_mask(width: int, height: int) -> np.ndarray:
    """255 where pixels are scene content; 0 on broadcast overlays that never move with the camera."""
    mask = np.full((height, width), 255, np.uint8)
    sx, sy = width / 1920, height / 1080
    cv2.rectangle(mask, (0, round(820 * sy)), (round(660 * sx), height), 0, -1)  # score bug and its tag
    cv2.rectangle(mask, (round(1620 * sx), 0), (width, round(190 * sy)), 0, -1)  # tournament logo
    return mask


@dataclass
class Reference:
    name: str
    corners: np.ndarray  # 4x2 singles corners in full-resolution reference pixels
    keypoints: np.ndarray  # Nx2 full-resolution coordinates
    descriptors: np.ndarray
    size: tuple[int, int]


def _features(frame: np.ndarray, detector, scale_to: int = MATCH_WIDTH):
    height, width = frame.shape[:2]
    scale = scale_to / width
    small = cv2.resize(frame, (scale_to, round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(graphics_mask(width, height), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    points = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2) / scale
    return points, descriptors


def make_detector():
    return cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02)


def build_reference(name: str, frame: np.ndarray, corners, detector=None) -> Reference:
    detector = detector or make_detector()
    points, descriptors = _features(frame, detector)
    if descriptors is None or len(points) < 100:
        raise ValueError("The reference frame has too few stable features; choose a sharper wide view.")
    height, width = frame.shape[:2]
    return Reference(name, np.asarray(corners, np.float32), points, descriptors, (width, height))


def court_lines(corners: np.ndarray, samples: int = 60):
    """Sample points along the singles boundary, service lines, center service line and net line."""
    to_image = cv2.getPerspectiveTransform(np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32), corners.astype(np.float32))
    segments = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 0), (0, 1)), ((1, 0), (1, 1))]
    segments += [((0, y), (1, y)) for y in SERVICE_FRACTIONS]
    segments += [((0.5, SERVICE_FRACTIONS[0]), (0.5, SERVICE_FRACTIONS[1]))]
    points = []
    for (x0, y0), (x1, y1) in segments:
        t = np.linspace(0.04, 0.96, samples)
        points.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))
    court = np.concatenate(points).astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(court, to_image).reshape(-1, 2)


def line_score(frame: np.ndarray, corners: np.ndarray) -> tuple[float, int]:
    """Fraction of sampled court-line points that sit on a thin bright line; (score, points checked)."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.blur(gray, (15, 15))
    bright = gray - background
    peak = cv2.dilate(bright, np.ones((7, 7), np.uint8))  # tolerate a few pixels of misregistration
    points = court_lines(corners)
    inside = (points[:, 0] >= 4) & (points[:, 0] < width - 4) & (points[:, 1] >= 4) & (points[:, 1] < height - 4)
    points = points[inside]
    if len(points) == 0:
        return 0.0, 0
    values = peak[points[:, 1].round().astype(int), points[:, 0].round().astype(int)]
    return float((values > 22).mean()), int(len(points))


def plausible(corners: np.ndarray, size: tuple[int, int]) -> bool:
    """Ordered convex court quadrilateral of sensible size, even when partly outside the frame."""
    if not np.isfinite(corners).all():
        return False
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    area = abs(cv2.contourArea(contour))
    frame_area = size[0] * size[1]
    near, far = corners[:2], corners[2:]
    return (
        cv2.isContourConvex(contour) and 0.02 * frame_area < area < 8 * frame_area
        and near[:, 1].mean() > far[:, 1].mean()  # near baseline is lower in the image
        and corners[0, 0] < corners[1, 0] and corners[3, 0] < corners[2, 0]
    )


@dataclass
class Match:
    corners: np.ndarray | None
    homography: np.ndarray | None
    inliers: int
    matches: int
    line_score: float
    line_points: int
    accepted: bool
    reason: str


def match_reference(frame: np.ndarray, reference: Reference, detector=None, matcher=None) -> Match:
    detector = detector or make_detector()
    matcher = matcher or cv2.BFMatcher(cv2.NORM_L2)
    points, descriptors = _features(frame, detector)
    if descriptors is None or len(points) < MIN_INLIERS:
        return Match(None, None, 0, 0, 0.0, 0, False, "too_few_features")
    pairs = matcher.knnMatch(reference.descriptors, descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return Match(None, None, 0, len(good), 0.0, 0, False, "too_few_matches")
    source = reference.keypoints[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    target = points[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PIXELS * (frame.shape[1] / MATCH_WIDTH))
    if homography is None:
        return Match(None, None, 0, len(good), 0.0, 0, False, "no_homography")
    inliers = int(inlier_mask.sum())
    corners = cv2.perspectiveTransform(reference.corners.reshape(-1, 1, 2), homography).reshape(4, 2)
    height, width = frame.shape[:2]
    if inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return Match(corners, homography, inliers, len(good), 0.0, 0, False, "weak_geometry")
    if not plausible(corners, (width, height)):
        return Match(corners, homography, inliers, len(good), 0.0, 0, False, "implausible_court")
    score, checked = line_score(frame, corners)
    if checked < MIN_LINE_POINTS:
        return Match(corners, homography, inliers, len(good), score, checked, False, "court_mostly_outside_frame")
    if score < MIN_LINE_SCORE:
        return Match(corners, homography, inliers, len(good), score, checked, False, "court_lines_not_found")
    return Match(corners, homography, inliers, len(good), score, checked, True, "ok")


LIBRARY_VERSION = 1
SCAN_STEP_SECONDS = 0.5
MAX_GAP_STEPS = 1.5


def read_library_entries(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != LIBRARY_VERSION:
        raise ValueError("Unsupported camera library version.")
    return data["references"]


def write_library_entries(path: Path, entries: list[dict]) -> None:
    names = [e["name"] for e in entries]
    if len(names) != len(set(names)):
        raise ValueError("Camera reference names must be unique.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": LIBRARY_VERSION, "references": entries}, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_references(entries: list[dict], project: dict, detector=None) -> list[Reference]:
    """Rebuild reference features from the source frames recorded in the library.

    An entry may carry the video's `path` directly; older entries name a video in the project instead.
    """
    detector = detector or make_detector()
    references = []
    for entry in entries:
        path = entry.get("path")
        if not path:
            video = project["videos"].get(entry["video_id"])
            if video is None:
                raise ValueError(f"Camera reference {entry['name']} points to a video that is no longer in the project.")
            path = video["path"]
        frame = read_frame(path, entry["frame"])
        references.append(build_reference(entry["name"], frame, entry["corners"], detector))
    return references


def best_match(frame: np.ndarray, references: list[Reference], detector, matcher) -> tuple[Match, str | None]:
    best, name = None, None
    for reference in references:
        match = match_reference(frame, reference, detector, matcher)
        if best is None or (match.accepted, match.inliers) > (best.accepted, best.inliers):
            best, name = match, reference.name
    if best is None:
        raise ValueError("Add at least one camera reference first.")
    return best, (name if best.accepted else None)


def scan_clip(path, references: list[Reference], fps: float, frames: int, progress=None) -> list[dict]:
    """Match every 0.5 s of a clip; returns one record per sampled frame (accepted or not)."""
    step = max(1, round(SCAN_STEP_SECONDS * fps))
    detector, matcher = make_detector(), cv2.BFMatcher(cv2.NORM_L2)
    cap = cv2.VideoCapture(str(path))
    samples = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        index = 0
        while index < frames:
            ok, frame = cap.read()
            if not ok:
                break
            match, name = best_match(frame, references, detector, matcher)
            samples.append({
                "frame": index, "accepted": match.accepted, "reference": name, "reason": match.reason,
                "inliers": match.inliers, "line_score": round(match.line_score, 3),
                "corners": match.corners.round(2).tolist() if match.accepted else None,
            })
            if progress and len(samples) % 50 == 0:
                progress(len(samples), index)
            # Skip (decode without converting) up to the next sampled frame.
            for _ in range(step - 1):
                if not cap.grab():
                    break
            index += step
    finally:
        cap.release()
    return samples


class CameraTrack:
    """Court corners over time, linearly interpolated between accepted samples of one continuous view."""

    def __init__(self, samples: list[dict], fps: float):
        self.fps = fps
        self.samples = samples
        self.step = max(1, round(SCAN_STEP_SECONDS * fps))
        self.segments: list[dict] = []
        run: list[dict] = []
        for sample in samples:
            if not sample["accepted"]:
                continue
            same_view = run and run[-1]["reference"] == sample["reference"]
            if run and same_view and sample["frame"] - run[-1]["frame"] <= MAX_GAP_STEPS * self.step:
                run.append(sample)
            else:
                self._close(run)
                run = [sample]
        self._close(run)
        self._starts = [s["start"] for s in self.segments]

    def _close(self, run):
        if run:
            self.segments.append({
                "start": run[0]["frame"], "end": run[-1]["frame"], "reference": run[0]["reference"],
                "points": [(s["frame"], np.array(s["corners"], float)) for s in run],
            })

    def _segment(self, frame: int):
        index = bisect.bisect_right(self._starts, frame) - 1
        if index >= 0 and frame <= self.segments[index]["end"]:
            return self.segments[index]
        return None

    def in_court_view(self, frame: int) -> bool:
        return self._segment(frame) is not None

    def speed_at(self, frame: int) -> float | None:
        """Largest court-corner shift in pixels per frame around `frame`; None outside a court view."""
        before, after = self.corners_at(frame - 1), self.corners_at(frame + 1)
        if before is None or after is None:
            here = self.corners_at(frame)
            return None if here is None else 0.0
        return float(np.abs(after - before).max() / 2)

    def corners_at(self, frame: int) -> np.ndarray | None:
        segment = self._segment(frame)
        if segment is None:
            return None
        points = segment["points"]
        index = bisect.bisect_right([p[0] for p in points], frame) - 1
        if index >= len(points) - 1:
            return points[-1][1].copy()
        (f0, c0), (f1, c1) = points[index], points[index + 1]
        weight = (frame - f0) / (f1 - f0)
        return (1 - weight) * c0 + weight * c1

    def coverage(self, frames: int) -> dict:
        covered = sum(s["end"] - s["start"] + self.step for s in self.segments)
        return {
            "segments": len(self.segments), "court_view_frames": min(frames, covered),
            "fraction": min(1.0, covered / frames) if frames else 0.0,
        }

    def to_dict(self) -> dict:
        return {"fps": self.fps, "step_frames": self.step, "samples": self.samples}

    @classmethod
    def from_dict(cls, data: dict) -> "CameraTrack":
        return cls(data["samples"], data["fps"])
