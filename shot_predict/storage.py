import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .court import Court, ZONES
from .video import metadata

SHOT_SOURCES = ("manual", "auto_confirmed", "auto_corrected")
REVIEW_DECISIONS = ("rejected", "out", "skipped")


def read_project(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "videos": {}, "shots": [], "reviews": [], "audits": []}
    with path.open(encoding="utf-8") as handle:
        project = json.load(handle)
    if project.get("version") != 1:
        raise ValueError("Unsupported annotation version.")
    project.setdefault("reviews", [])
    project.setdefault("audits", [])
    return project


def write_project(path: Path, project: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(project, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def backup_project(path: Path, folder: Path | None = None, keep: int = 20) -> Path | None:
    """Copy the current project file once per distinct content; returns the backup path, if one was made."""
    if not path.is_file():
        return None
    folder = folder or path.parent / "backups"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    existing = sorted(folder.glob(f"{path.stem}-*-{digest}{path.suffix}"))
    if existing:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = folder / f"{path.stem}-{stamp}-{digest}{path.suffix}"
    shutil.copy2(path, target)
    for old in sorted(folder.glob(f"{path.stem}-*{path.suffix}"))[:-keep]:
        old.unlink(missing_ok=True)
    return target


def input_corners(shot: dict, video: dict) -> list[list[float]]:
    """Court corners that apply to a shot's pre-contact frames.

    Shots labeled before the camera library existed have no shot-level value and use the clip's calibration.
    """
    corners = shot.get("input_court_corners") or video.get("corners")
    if not corners:
        raise ValueError(f"Shot {shot.get('id')} has no court calibration for its input frames.")
    return corners


def video_id_for(path: Path) -> str:
    return hashlib.sha256(str(Path(path).expanduser().resolve()).encode()).hexdigest()[:16]


def add_video(
    project: dict, path: Path, match_id: str, near_player: str = "", far_player: str = "",
) -> str:
    match_id = match_id.strip()
    if not match_id:
        raise ValueError("A match ID is required.")
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Video is not a file: {path}")
    video_id = video_id_for(path)
    if video_id in project["videos"]:
        raise ValueError(f"Video is already imported: {path}")
    project["videos"][video_id] = {
        "path": str(path), "match_id": match_id,
        "near_player": near_player.strip(), "far_player": far_player.strip(),
        "corners": None, **metadata(path),
    }
    return video_id


def add_shot(
    project: dict, video_id: str, contact: int, bounce: int,
    hitter: str, shot_type: str, pixel: tuple[float, float],
    *, corners: list[list[float]] | None = None, input_corners_at_contact: list[list[float]] | None = None,
    source: str = "manual", proposal: dict | None = None, review_seconds: float | None = None,
) -> dict:
    """Add a label.

    corners: camera calibration at the bounce frame (defaults to the clip calibration), used to
        convert the clicked pixel to court coordinates.
    input_corners_at_contact: camera calibration for the pre-contact input frames when the camera
        moves; omitted for fixed-camera clips.
    """
    video = project["videos"][video_id]
    if not (0 <= contact < bounce < video["frames"]):
        raise ValueError("Contact must precede bounce and both must be inside the video.")
    if shot_type not in ("groundstroke", "serve"):
        raise ValueError("Supported shot types are groundstroke and serve.")
    if source not in SHOT_SOURCES:
        raise ValueError(f"Shot source must be one of {SHOT_SOURCES}.")
    if source != "manual" and not isinstance(proposal, dict):
        raise ValueError("Automatically proposed shots must keep their original proposal.")
    if review_seconds is not None and not (isinstance(review_seconds, (int, float)) and review_seconds >= 0):
        raise ValueError("Review time must be a nonnegative number of seconds.")
    bounce_corners = corners if corners is not None else video["corners"]
    if not bounce_corners:
        raise ValueError("Calibrate the court before labeling a shot.")
    court = Court(bounce_corners)
    if input_corners_at_contact is not None:
        Court(input_corners_at_contact)
    x, y = court.normalized(*pixel)
    zone = court.zone(x, y, hitter)
    if any(s["video_id"] == video_id and s["contact_frame"] == contact for s in project["shots"]):
        raise ValueError("This contact is already labeled. Delete its existing label first.")
    shot = {
        "id": uuid4().hex, "video_id": video_id, "contact_frame": int(contact),
        "bounce_frame": int(bounce), "hitter": hitter, "shot_type": shot_type,
        "landing": [x, y], "zone": zone,
        "landing_pixel": [float(pixel[0]), float(pixel[1])],
        "court_corners": [list(point) for point in bounce_corners],
    }
    if input_corners_at_contact is not None:
        shot["input_court_corners"] = [list(point) for point in input_corners_at_contact]
    if source != "manual":
        shot["source"] = source
        shot["proposal"] = proposal
        shot["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if review_seconds is not None:
        shot["review_seconds"] = float(review_seconds)
    if zone not in ZONES:
        raise ValueError("Unknown landing zone.")
    project["shots"].append(shot)
    return shot


def add_review(project: dict, video_id: str, proposal_id: str, decision: str) -> dict:
    """Remember a proposal that was rejected, called out, or skipped so it is not queued again."""
    if video_id not in project["videos"]:
        raise ValueError("Unknown video.")
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"Review decision must be one of {REVIEW_DECISIONS}.")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("A proposal ID is required.")
    reviews = project.setdefault("reviews", [])
    reviews[:] = [r for r in reviews if not (r["video_id"] == video_id and r["proposal_id"] == proposal_id)]
    review = {
        "video_id": video_id, "proposal_id": proposal_id, "decision": decision,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    reviews.append(review)
    return review


def add_audit(project: dict, video_id: str, proposal_id: str, caught: bool) -> dict:
    """Record how the reviewer handled a deliberately wrong card (an audit); it never creates a label."""
    if video_id not in project["videos"]:
        raise ValueError("Unknown video.")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("A proposal ID is required.")
    audit = {
        "video_id": video_id, "proposal_id": proposal_id, "caught": bool(caught),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    project.setdefault("audits", []).append(audit)
    return audit
