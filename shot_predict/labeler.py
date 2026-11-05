import math
import mimetypes
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.media_file_storage import MediaFileStorageError

from .autopilot import registry, review
from .court import Court
from .storage import add_shot
from .video import clip_indices

DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "data" / "analysis"
DEFAULT_CAMERA_LIBRARY = Path(__file__).resolve().parents[1] / "data" / "camera" / "references.json"

_component = components.declare_component(
    "shot_label_workspace", path=str(Path(__file__).parent / "labeler_frontend")
)


def video_url(path: str) -> str:
    if not st.runtime.exists():
        raise RuntimeError("Run the labeling app with Streamlit.")
    mimetype, _ = mimetypes.guess_type(path)
    if not mimetype or not mimetype.startswith("video/"):
        raise ValueError(f"Cannot determine the video format for {path}")
    # Components need the same session-scoped media URL that st.video uses.
    try:
        return st.runtime.get_instance().media_file_mgr.add(path, mimetype, "shot-lab-video")
    except MediaFileStorageError as exc:
        raise ValueError(f"Cannot load video {path}: {exc}") from exc


def court_grid(corners: list[list[float]] | None) -> list[list[list[int]]]:
    if corners is None:
        return []
    court = Court(corners)
    lines = []
    for x in (0, 1 / 3, 2 / 3, 1):
        lines.append([list(court.pixel(x, 0)), list(court.pixel(x, 1))])
    for y in (0, 0.25, 0.5, 0.75, 1):
        lines.append([list(court.pixel(0, y)), list(court.pixel(1, y))])
    return lines


def workspace(**kwargs):
    return _component(**kwargs, key="shot-workspace", default=None)


def _frame(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole frame number.")
    return value


def _point(value, video: dict) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("A video position needs an x and y coordinate.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("Video coordinates must be finite numbers.")
    x, y = value
    if not (0 <= x < video["width"] and 0 <= y < video["height"]):
        raise ValueError("Click inside the video image.")
    return [float(x), float(y)]


def apply_action(project: dict, video_id: str, event: dict, analysis_root: Path | None = None,
                 camera_library: Path | None = None) -> dict:
    analysis_root = Path(analysis_root) if analysis_root else DEFAULT_ANALYSIS_ROOT
    if video_id not in project["videos"] or event.get("video_id") != video_id:
        raise ValueError("The clip changed. Please try the action again.")
    video = project["videos"][video_id]
    action = event.get("action")
    if action == "validate_batch":
        # Each verdict names its own clip, so verdicts still unsent when the reviewer moved on are applied correctly.
        return review.validate_batches(project, event, analysis_root, _point, _frame)
    if action == "save_camera_reference":
        raw_points = event.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            raise ValueError("Click all four singles baseline corners.")
        points = [_point(point, video) for point in raw_points]
        Court(points)
        result = registry.register_reference(
            video, _frame(event.get("frame"), "Frame"), points, Path(camera_library or DEFAULT_CAMERA_LIBRARY))
        return {"message": f"Camera registered as {result['name']}. Run the batch to analyze every clip from this camera."}
    if action == "save_calibration":
        raw_points = event.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            raise ValueError("Click all four singles baseline corners.")
        points = [_point(point, video) for point in raw_points]
        Court(points)
        video["corners"] = points
        return {"message": "Court saved. Check the green lines when the camera moves."}
    if action == "save_shot":
        track_present = review.load_track(analysis_root, video_id) is not None
        if video["corners"] is None and not track_present:
            raise ValueError("Calibrate the court before labeling a shot.")
        contact = _frame(event.get("contact_frame"), "Contact")
        bounce = _frame(event.get("bounce_frame"), "Bounce")
        pixel = _point(event.get("landing_pixel"), video)
        clip_indices(contact, video["fps"], 0.6, 8, 300)
        # Moving-camera clips take the court from the camera track at each frame; fixed clips keep the clip calibration.
        bounce_corners = contact_corners = None
        if track_present:
            bounce_corners = review.corners_for_frame(project, video_id, analysis_root, bounce)
            contact_corners = review.corners_for_frame(project, video_id, analysis_root, contact)
        shot = add_shot(
            project, video_id, contact, bounce,
            event.get("hitter"), event.get("shot_type"), tuple(pixel),
            corners=bounce_corners, input_corners_at_contact=contact_corners,
        )
        return {"message": f"Saved: {shot['hitter']} - {shot['zone'].replace('_', ' ')}"}
    if action == "accept_proposal":
        clip_indices(_frame(event.get("contact_frame"), "Contact"), video["fps"], 0.6, 8, 300)
        return review.accept_proposal(project, video_id, event, analysis_root, _point, _frame)
    if action == "reject_proposal":
        return review.reject_proposal(project, video_id, event, analysis_root)
    if action == "undo_review":
        return review.undo_review(project, video_id, event)
    if action == "undo":
        shots = [shot for shot in project["shots"] if shot["video_id"] == video_id]
        if not shots or shots[-1]["id"] != event.get("shot_id"):
            raise ValueError("The last label changed. Refresh and try again.")
        shot_id = shots[-1]["id"]
        project["shots"] = [shot for shot in project["shots"] if shot["id"] != shot_id]
        return {"message": "Last label removed."}
    raise ValueError(f"Unknown labeling action: {action}")
