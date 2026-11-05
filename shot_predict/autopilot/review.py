"""Review-queue backend: loads proposals for the labeling UI and turns confirmed ones into labels.

Proposals come from the offline analysis (never from the browser): the saved shot always records the
original proposal as read from disk, and provenance says whether the human changed anything.
"""

import json
import math
from pathlib import Path

from ..court import ZONES
from ..storage import add_audit, add_review, add_shot
from . import inbox
from .camera import CameraTrack

MAX_QUEUE = 500
MAX_TRAIL_POINTS = 600
SAME_SHOT_SECONDS = 0.25
LANDING_TOLERANCE_PIXELS = 0.5


def clip_folder(analysis_root: Path, video_id: str) -> Path:
    return Path(analysis_root) / video_id


def load_track(analysis_root: Path, video_id: str) -> CameraTrack | None:
    path = clip_folder(analysis_root, video_id) / "camera.json"
    if not path.is_file():
        return None
    try:
        return CameraTrack.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, OSError, TypeError):
        return None


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _corners(value) -> bool:
    return (
        isinstance(value, list) and len(value) == 4
        and all(isinstance(p, list) and len(p) == 2 and all(_finite(v) for v in p) for p in value)
    )


def _frame(value, frames: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < frames


def clean_proposal(raw, frames: int) -> dict | None:
    """Validated copy of one proposal, or None when it is malformed."""
    try:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
            return None
        contact, bounce = raw.get("contact_frame"), raw.get("bounce_frame")
        if not _frame(contact, frames) or raw.get("hitter") not in ("near", "far"):
            return None
        if bounce is not None and not (_frame(bounce, frames) and bounce > contact):
            return None
        if raw.get("shot_type") not in ("groundstroke", "serve"):
            return None
        for key in ("confidence", "bounce_confidence", "hitter_confidence"):
            if not _finite(raw.get(key, 0.0)) or not 0 <= raw.get(key, 0.0) <= 1:
                return None
        zone, pixel = raw.get("zone"), raw.get("landing_pixel")
        if zone is not None and zone not in ZONES:
            return None
        if pixel is not None and not (isinstance(pixel, list) and len(pixel) == 2 and all(_finite(v) for v in pixel)):
            return None
        for key in ("contact_corners", "bounce_corners"):
            if raw.get(key) is not None and not _corners(raw[key]):
                return None
        players = raw.get("players") or {}
        cleaned_players = {}
        for side in ("near", "far"):
            box = players.get(side)
            if box is not None and not (isinstance(box, list) and len(box) >= 4 and all(_finite(v) for v in box[:4])):
                return None
            cleaned_players[side] = None if box is None else [float(v) for v in box[:4]]
        trail = [
            [int(p[0]), float(p[1]), float(p[2])]
            for p in (raw.get("ball_trail") or [])[:MAX_TRAIL_POINTS]
            if isinstance(p, list) and len(p) == 3 and all(_finite(v) for v in p)
        ]
        return {
            "id": raw["id"], "contact_frame": contact, "hitter": raw["hitter"],
            "hitter_confidence": float(raw.get("hitter_confidence", raw.get("confidence", 0.0))),
            "shot_type": raw["shot_type"], "confidence": float(raw.get("confidence", 0.0)),
            "bounce_frame": bounce, "bounce_confidence": float(raw.get("bounce_confidence", 0.0)),
            "landing_pixel": None if pixel is None else [float(pixel[0]), float(pixel[1])],
            "zone": zone, "contact_corners": raw.get("contact_corners"), "bounce_corners": raw.get("bounce_corners"),
            "players": cleaned_players, "ball_trail": trail,
            "reasons": [r for r in (raw.get("reasons") or []) if isinstance(r, str)][:12],
        }
    except (TypeError, ValueError, AttributeError):
        return None


def read_proposals(analysis_root: Path, video_id: str, frames: int) -> tuple[list[dict], int, str]:
    """(valid proposals, number dropped as malformed, message)."""
    path = clip_folder(analysis_root, video_id) / "proposals.json"
    if not path.is_file():
        return [], 0, "No analysis for this clip yet. Run: python -m shot_predict.autopilot.cli analyze"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data["proposals"]
        if not isinstance(raw, list):
            raise ValueError("proposals must be a list")
    except (ValueError, KeyError, OSError, TypeError):
        return [], 0, "The analysis file for this clip is unreadable. Re-run the analysis."
    cleaned = [clean_proposal(p, frames) for p in raw]
    valid = [p for p in cleaned if p is not None]
    dropped = len(raw) - len(valid)  # malformed only; harmless duplicates are merged silently below
    seen, unique = set(), []
    for p in valid:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    return unique, dropped, ""


def load_review(project: dict, video_id: str, analysis_root: Path) -> dict:
    """Payload for the UI: the unreviewed queue, camera samples and session statistics."""
    video = project["videos"][video_id]
    proposals, dropped, message = read_proposals(analysis_root, video_id, video["frames"])
    shots = [s for s in project["shots"] if s["video_id"] == video_id]
    window = max(1, round(SAME_SHOT_SECONDS * video["fps"]))
    reviewed = {r["proposal_id"] for r in project.get("reviews", []) if r["video_id"] == video_id}
    queue = [
        p for p in proposals
        if p["id"] not in reviewed and not any(abs(p["contact_frame"] - s["contact_frame"]) <= window for s in shots)
    ]
    queue.sort(key=lambda p: p["contact_frame"])
    for p in queue:
        p["validatable"] = inbox.is_validatable(p, video["fps"])
    track = load_track(analysis_root, video_id)
    camera = None
    if track is not None:
        camera = {
            "step_frames": track.step,
            "samples": [
                {"frame": s["frame"], "reference": s["reference"], "corners": s["corners"]}
                for s in track.samples if s["accepted"]
            ],
        }
    from_proposals = [s for s in shots if s.get("source") in ("auto_confirmed", "auto_corrected")]
    seconds = [s["review_seconds"] for s in from_proposals if "review_seconds" in s]
    return {
        "queue": queue[:MAX_QUEUE], "queue_total": len(queue), "camera": camera, "dropped": dropped, "message": message,
        "inbox": inbox.summarize(project, analysis_root, video_id),
        "audit": audit_summary(project),
        "stats": {
            "proposals": len(proposals), "saved_from_proposals": len(from_proposals),
            "auto_confirmed": sum(s["source"] == "auto_confirmed" for s in from_proposals),
            "auto_corrected": sum(s["source"] == "auto_corrected" for s in from_proposals),
            "rejected": len(reviewed),
            "mean_review_seconds": sum(seconds) / len(seconds) if seconds else None,
        },
    }


def corners_for_frame(project: dict, video_id: str, analysis_root: Path, frame: int):
    """Court corners for a frame: the camera track when the clip has one, otherwise None (use the clip calibration)."""
    video = project["videos"][video_id]
    track = load_track(analysis_root, video_id)
    if track is None:
        if not video.get("corners"):
            raise ValueError("Calibrate the court before labeling a shot.")
        return None
    corners = track.corners_at(frame)
    if corners is None:
        raise ValueError(
            "This frame is not in a recognized court-view segment. Move to a frame where the court is visible."
        )
    return [[float(v) for v in point] for point in corners]


def _same_point(a, b) -> bool:
    return a is not None and b is not None and all(abs(x - y) <= LANDING_TOLERANCE_PIXELS for x, y in zip(a, b))


def accept_proposal(
    project: dict, video_id: str, event: dict, analysis_root: Path, point, frame_checker, proposals: list | None = None,
) -> dict:
    """Save a proposal (with the human's edits) as a label. `point` validates the clicked/kept landing pixel."""
    video = project["videos"][video_id]
    if proposals is None:
        proposals, _, _ = read_proposals(analysis_root, video_id, video["frames"])
    proposal = next((p for p in proposals if p["id"] == event.get("proposal_id")), None)
    if proposal is None:
        raise ValueError("That proposal is no longer available. Refresh the page.")
    contact = frame_checker(event.get("contact_frame"), "Contact")
    bounce = frame_checker(event.get("bounce_frame"), "Bounce")
    landing = point(event.get("landing_pixel"), video)
    hitter, shot_type = event.get("hitter"), event.get("shot_type")
    edited = []
    if contact != proposal["contact_frame"]:
        edited.append("contact_frame")
    if bounce != proposal["bounce_frame"]:
        edited.append("bounce_frame")
    if hitter != proposal["hitter"]:
        edited.append("hitter")
    if shot_type != proposal["shot_type"]:
        edited.append("shot_type")
    if not _same_point(landing, proposal["landing_pixel"]):
        edited.append("landing")
    seconds = event.get("review_seconds")
    if seconds is not None and not (_finite(seconds) and 0 <= seconds <= 3600):
        raise ValueError("Review time is invalid.")
    track = load_track(analysis_root, video_id)
    bounce_corners = contact_corners = None
    if track is not None:
        bounce_corners = corners_for_frame(project, video_id, analysis_root, bounce)
        contact_corners = corners_for_frame(project, video_id, analysis_root, contact)
    elif not video.get("corners"):
        raise ValueError("Calibrate the court before labeling a shot.")
    stored = {k: v for k, v in proposal.items() if k != "ball_trail"}
    shot = add_shot(
        project, video_id, contact, bounce, hitter, shot_type, tuple(landing),
        corners=bounce_corners, input_corners_at_contact=contact_corners,
        source="auto_corrected" if edited else "auto_confirmed", proposal=stored, review_seconds=seconds,
    )
    shot["edited_fields"] = edited
    return {"message": f"Saved: {shot['hitter']} - {shot['zone'].replace('_', ' ')}" + (" (edited)" if edited else ""),
            "shot_id": shot["id"], "edited_fields": edited}


def reject_proposal(project: dict, video_id: str, event: dict, analysis_root: Path) -> dict:
    proposals, _, _ = read_proposals(analysis_root, video_id, project["videos"][video_id]["frames"])
    if not any(p["id"] == event.get("proposal_id") for p in proposals):
        raise ValueError("That proposal is no longer available. Refresh the page.")
    decision = event.get("decision")
    add_review(project, video_id, event["proposal_id"], decision)
    return {"message": {"rejected": "Marked as not a shot.", "out": "Marked as out or net.",
                        "skipped": "Skipped."}[decision]}


def undo_review(project: dict, video_id: str, event: dict) -> dict:
    reviews = project.get("reviews", [])
    for index in range(len(reviews) - 1, -1, -1):
        r = reviews[index]
        if r["video_id"] == video_id and r["proposal_id"] == event.get("proposal_id"):
            del reviews[index]
            return {"message": "Review decision removed; the proposal is back in the queue."}
    raise ValueError("That review decision no longer exists.")


MAX_BATCH = 60
VERDICTS = ("correct", "wrong", "skip")


def audit_summary(project: dict) -> dict:
    audits = project.get("audits", [])
    caught = sum(1 for a in audits if a.get("caught"))
    return {"shown": len(audits), "caught": caught, "missed": len(audits) - caught}


def validate_batches(project: dict, event: dict, analysis_root: Path, point, frame_checker) -> dict:
    """validate_batch across clips: every decision names its own clip, so unsent verdicts survive a clip switch."""
    decisions = event.get("decisions")
    if not isinstance(decisions, list) or not 0 < len(decisions) <= MAX_BATCH:
        raise ValueError(f"Send between 1 and {MAX_BATCH} decisions at a time.")
    by_clip: dict[str, list] = {}
    for position, decision in enumerate(decisions):
        clip = (decision.get("video_id") or event.get("video_id")) if isinstance(decision, dict) else None
        by_clip.setdefault(clip, []).append((position, decision))
    results: list = [None] * len(decisions)
    saved = 0
    for clip, items in by_clip.items():
        if clip not in project["videos"]:
            for position, decision in items:
                results[position] = {"proposal_id": decision.get("proposal_id") if isinstance(decision, dict) else None,
                                     "video_id": clip if isinstance(clip, str) else None, "ok": False, "message": "Unknown clip."}
            continue
        outcome = validate_batch(project, clip, {"decisions": [d for _, d in items]}, analysis_root, point, frame_checker)
        saved += outcome["saved"]
        for (position, _), result in zip(items, outcome["results"]):
            results[position] = {**result, "video_id": clip}
    failed = sum(1 for r in results if not r["ok"])
    return {
        "message": f"Saved {saved} labels" + (f"; {failed} could not be applied." if failed else "."),
        "results": results, "saved": saved, "failed": failed,
    }


def validate_batch(project: dict, video_id: str, event: dict, analysis_root: Path, point, frame_checker) -> dict:
    """Apply a batch of one-look verdicts. Labels always come from the proposal on disk, never from the browser.

    Items fail independently (a clip can change under the reviewer); the result lists each outcome.
    verdicts: "correct" saves the proposal unchanged (source auto_confirmed), "wrong" and "skip" are remembered so the
    proposal is not queued again. An item with kind "audit" records whether a deliberately wrong card was caught.
    """
    decisions = event.get("decisions")
    if not isinstance(decisions, list) or not 0 < len(decisions) <= MAX_BATCH:
        raise ValueError(f"Send between 1 and {MAX_BATCH} decisions at a time.")
    video = project["videos"][video_id]
    proposals, _, _ = read_proposals(analysis_root, video_id, video["frames"])
    by_id = {p["id"]: p for p in proposals}
    results, saved = [], 0
    for decision in decisions:
        proposal_id = decision.get("proposal_id") if isinstance(decision, dict) else None
        try:
            if proposal_id not in by_id:
                raise ValueError("That proposal is no longer available.")
            if decision.get("kind") == "audit":
                add_audit(project, video_id, proposal_id, bool(decision.get("caught")))
                results.append({"proposal_id": proposal_id, "ok": True})
                continue
            verdict = decision.get("verdict")
            if verdict not in VERDICTS:
                raise ValueError("Unknown verdict.")
            if verdict == "correct":
                p = by_id[proposal_id]
                if not inbox.is_validatable(p, video["fps"]):
                    raise ValueError("This proposal has no landing to confirm; use the review flow.")
                saved_shot = accept_proposal(
                    project, video_id,
                    {"proposal_id": proposal_id, "contact_frame": p["contact_frame"], "bounce_frame": p["bounce_frame"],
                     "hitter": p["hitter"], "shot_type": p["shot_type"], "landing_pixel": p["landing_pixel"],
                     "review_seconds": decision.get("review_seconds")},
                    analysis_root, point, frame_checker, proposals=proposals,
                )
                results.append({"proposal_id": proposal_id, "ok": True, "shot_id": saved_shot["shot_id"]})
                saved += 1
            else:
                add_review(project, video_id, proposal_id, "rejected" if verdict == "wrong" else "skipped")
                results.append({"proposal_id": proposal_id, "ok": True})
        except ValueError as exc:
            results.append({"proposal_id": proposal_id if isinstance(proposal_id, str) else None, "ok": False, "message": str(exc)})
    failed = sum(1 for r in results if not r["ok"])
    return {
        "message": f"Saved {saved} labels" + (f"; {failed} could not be applied." if failed else "."),
        "results": results, "saved": saved, "failed": failed,
    }
