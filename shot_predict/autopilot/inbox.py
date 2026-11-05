"""The validation inbox: which proposals can be confirmed at a glance, and where they are across all clips.

A proposal is *validatable* when it carries a predicted landing zone with a plausible contact-to-bounce time.
Measured on labeled Turin rallies, such proposals were fully right (hitter, zone and contact within 250 ms) 22 of
27 times, while proposals without a landing zone can never be confirmed without a click. The confidence scores do not
separate right from wrong beyond that (81-86% at every threshold), so there is deliberately no finer strictness knob.

Every analyzed clip gets a small `inbox.json` next to its `proposals.json`, so the app can summarize hundreds of clips
without parsing their (large) proposal files.
"""

import json
from pathlib import Path

from ..court import ZONES
from ..video import clip_indices

INBOX_VERSION = 1
MIN_INTERVAL_SECONDS = 0.25
MAX_INTERVAL_SECONDS = 1.6
SAME_SHOT_SECONDS = 0.25
MAX_LISTED_CLIPS = 400


def is_validatable(proposal: dict, fps: float) -> bool:
    bounce, contact = proposal.get("bounce_frame"), proposal.get("contact_frame")
    pixel = proposal.get("landing_pixel")
    if proposal.get("zone") not in ZONES or not pixel or bounce is None or contact is None or fps <= 0:
        return False
    try:
        clip_indices(contact, fps, 0.6, 8, 300)  # a label needs this much video before contact for the model input
    except ValueError:
        return False
    return MIN_INTERVAL_SECONDS <= (bounce - contact) / fps <= MAX_INTERVAL_SECONDS


def _source(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_inbox(proposals_path: Path) -> dict | None:
    """Compact summary of one clip's proposals, or None when the file is missing or unreadable."""
    try:
        data = json.loads(proposals_path.read_text(encoding="utf-8"))
        fps = float(data["fps"])
        items = [
            {"id": p["id"], "contact_frame": int(p["contact_frame"]), "validatable": is_validatable(p, fps)}
            for p in data["proposals"] if isinstance(p, dict) and isinstance(p.get("id"), str)
        ]
        return {"version": INBOX_VERSION, "source": _source(proposals_path), "fps": fps, "items": items}
    except (OSError, ValueError, KeyError, TypeError):
        return None


_CACHE: dict[str, tuple[tuple, dict]] = {}


def clip_inbox(analysis_root: Path, video_id: str) -> dict | None:
    """The clip's inbox, rebuilt (and rewritten, best effort) when its proposals changed since it was written."""
    folder = Path(analysis_root) / video_id
    proposals_path, inbox_path = folder / "proposals.json", folder / "inbox.json"
    try:
        stat = proposals_path.stat()
    except OSError:
        return None
    stamp = (stat.st_size, stat.st_mtime_ns)
    cached = _CACHE.get(str(inbox_path))
    if cached and cached[0] == stamp:
        return cached[1]
    inbox = None
    try:
        saved = json.loads(inbox_path.read_text(encoding="utf-8"))
        if saved.get("version") == INBOX_VERSION and saved.get("source") == {"size": stamp[0], "mtime_ns": stamp[1]}:
            inbox = saved
    except (OSError, ValueError, TypeError):
        pass
    if inbox is None:
        inbox = build_inbox(proposals_path)
        if inbox is None:
            return None
        write_inbox(folder, inbox)
    _CACHE[str(inbox_path)] = (stamp, inbox)
    return inbox


def write_inbox(folder: Path, inbox: dict | None = None) -> None:
    inbox = inbox or build_inbox(folder / "proposals.json")
    if inbox is None:
        return
    temporary = folder / "inbox.json.tmp"
    try:
        temporary.write_text(json.dumps(inbox), encoding="utf-8")
        temporary.replace(folder / "inbox.json")
    except OSError:
        temporary.unlink(missing_ok=True)


def decided_ids(project: dict) -> dict[str, set[str]]:
    decided: dict[str, set[str]] = {}
    for review in project.get("reviews", []):
        decided.setdefault(review["video_id"], set()).add(review["proposal_id"])
    return decided


def pending_items(inbox: dict, decided: set[str], labeled_contacts: list[int], fps: float) -> list[dict]:
    """Validatable proposals not yet decided and not already labeled (same-shot window)."""
    window = max(1, round(SAME_SHOT_SECONDS * fps))
    return [
        item for item in inbox["items"]
        if item["validatable"] and item["id"] not in decided
        and not any(abs(item["contact_frame"] - c) <= window for c in labeled_contacts)
    ]


def summarize(project: dict, analysis_root: Path, current: str | None = None) -> dict:
    """Where the validation work is: clips (with their position in watching order) that still have pending cards.

    `current_order` is the open clip's position, so the UI can continue after it even once it has no cards left.
    """
    decided = decided_ids(project)
    contacts: dict[str, list[int]] = {}
    for shot in project["shots"]:
        contacts.setdefault(shot["video_id"], []).append(shot["contact_frame"])
    clips, analyzed, pending_total, current_order = [], 0, 0, -1
    ordered = sorted(project["videos"].items(), key=lambda kv: (kv[1]["match_id"], kv[1]["path"]))
    for order, (video_id, video) in enumerate(ordered):
        if video_id == current:
            current_order = order
        inbox = clip_inbox(analysis_root, video_id)
        if inbox is None:
            continue
        analyzed += 1
        pending = len(pending_items(inbox, decided.get(video_id, set()), contacts.get(video_id, []), video["fps"]))
        pending_total += pending
        if pending:
            clips.append({"video_id": video_id, "pending": pending, "order": order})
    return {
        "clips": clips[:MAX_LISTED_CLIPS], "clips_with_cards": len(clips), "pending_total": pending_total,
        "analyzed_clips": analyzed, "total_clips": len(project["videos"]), "current_order": current_order,
    }
