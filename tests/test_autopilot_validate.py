import json

import pytest

from shot_predict import labeler
from shot_predict.autopilot import inbox, review
from shot_predict.court import Court

# The synthetic clips are 80 frames at 20 fps; a label needs 18+ frames of pre-contact context.
CORNERS = [[20, 110], [140, 110], [140, 10], [20, 10]]
COURT = Court(CORNERS)


def proposal(contact, bounce=None, hitter="near", zone="ad_deep", **extra):
    return {
        "id": f"training-{contact}", "contact_frame": contact, "hitter": hitter, "hitter_confidence": 0.8,
        "shot_type": "groundstroke", "confidence": 0.8,
        "bounce_frame": bounce, "bounce_confidence": 0.4,
        "landing_pixel": list(COURT.target(zone, hitter)) if zone else None, "zone": zone,
        "contact_corners": CORNERS, "bounce_corners": CORNERS,
        "players": {"near": [50, 70, 70, 108], "far": None}, "ball_trail": [[contact, 60.0, 80.0], [contact + 1, 61.0, 78.0]],
        "reasons": [], **extra,
    }


def write_analysis(root, proposals, video_id="training"):
    folder = root / video_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "proposals.json").write_text(
        json.dumps({"version": "t", "fps": 20.0, "frames": 80, "proposals": proposals}), encoding="utf-8")


def batch(*decisions):
    return {"video_id": "training", "action": "validate_batch", "decisions": list(decisions)}


def test_only_proposals_with_a_plausible_landing_and_enough_context_are_validatable():
    fps = 20.0
    assert inbox.is_validatable(proposal(20, 28), fps)
    assert not inbox.is_validatable(proposal(20, 28, zone=None), fps)  # nothing to confirm
    assert not inbox.is_validatable({**proposal(20, 28), "landing_pixel": None}, fps)
    assert not inbox.is_validatable(proposal(20, None), fps)
    assert not inbox.is_validatable(proposal(20, 22), fps)  # bounces 0.1 s after contact: implausible
    assert not inbox.is_validatable(proposal(20, 60), fps)  # 2 s later: not this shot's bounce
    assert not inbox.is_validatable(proposal(10, 18), fps)  # no pre-contact frames to build the model input from


def test_a_batch_confirms_proposals_unchanged_and_ignores_anything_the_browser_adds(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28), proposal(62, 70, hitter="far", zone="deuce_deep"), proposal(40, 48, zone=None)])
    shots_before = len(project["shots"])
    result = labeler.apply_action(project, "training", batch(
        # The browser tries to smuggle in a different hitter, landing and frame: none of it is read.
        {"proposal_id": "training-20", "verdict": "correct", "review_seconds": 2.5, "hitter": "far",
         "landing_pixel": [1, 1], "contact_frame": 33},
        {"proposal_id": "training-62", "verdict": "wrong"},
        {"proposal_id": "training-40", "verdict": "correct"},
        {"proposal_id": "training-missing", "verdict": "correct"},
        {"proposal_id": "training-20", "verdict": "bogus"},
    ), root)
    assert result["saved"] == 1 and result["failed"] == 3
    ok, rejected, no_landing, missing, bogus = result["results"]
    assert ok["ok"] and ok["shot_id"] and rejected["ok"]
    assert not no_landing["ok"] and "landing" in no_landing["message"]
    assert not missing["ok"] and not bogus["ok"]
    shot = project["shots"][-1]
    assert len(project["shots"]) == shots_before + 1 and shot["id"] == ok["shot_id"]
    assert (shot["contact_frame"], shot["bounce_frame"], shot["hitter"], shot["zone"]) == (20, 28, "near", "ad_deep")
    assert shot["source"] == "auto_confirmed" and shot["edited_fields"] == [] and shot["review_seconds"] == 2.5
    assert [(r["proposal_id"], r["decision"]) for r in project["reviews"]] == [("training-62", "rejected")]


def test_a_confirmed_proposal_cannot_be_confirmed_twice_and_a_skip_is_remembered(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28), proposal(62, 70)])
    labeler.apply_action(project, "training", batch({"proposal_id": "training-20", "verdict": "correct"}), root)
    again = labeler.apply_action(project, "training", batch(
        {"proposal_id": "training-20", "verdict": "correct"}, {"proposal_id": "training-62", "verdict": "skip"}), root)
    assert again["saved"] == 0 and again["failed"] == 1 and "already labeled" in again["results"][0]["message"]
    assert [(r["proposal_id"], r["decision"]) for r in project["reviews"]] == [("training-62", "skipped")]


def test_one_batch_can_span_clips_and_an_unknown_clip_fails_alone(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28)], video_id="training")
    write_analysis(root, [{**proposal(20, 28), "id": "validation-20"}], video_id="validation")
    result = labeler.apply_action(project, "training", batch(
        {"video_id": "validation", "proposal_id": "validation-20", "verdict": "correct"},
        {"proposal_id": "training-20", "verdict": "correct"},  # no clip named: the current one
        {"video_id": "elsewhere", "proposal_id": "x", "verdict": "correct"},
    ), root)
    assert (result["saved"], result["failed"]) == (2, 1)
    assert [(r["video_id"], r["ok"]) for r in result["results"]] == [("validation", True), ("training", True), ("elsewhere", False)]
    saved = {s["video_id"] for s in project["shots"] if s.get("source") == "auto_confirmed"}
    assert saved == {"training", "validation"}


def test_audit_results_are_recorded_without_creating_labels(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28), proposal(62, 70)])
    shots_before = len(project["shots"])
    result = labeler.apply_action(project, "training", batch(
        {"kind": "audit", "proposal_id": "training-20", "caught": True},
        {"kind": "audit", "proposal_id": "training-62", "caught": False}), root)
    assert result["failed"] == 0 and len(project["shots"]) == shots_before and not project.get("reviews")
    assert review.audit_summary(project) == {"shown": 2, "caught": 1, "missed": 1}
    assert review.load_review(project, "training", root)["audit"] == {"shown": 2, "caught": 1, "missed": 1}


@pytest.mark.parametrize("decisions", [[], "nope", None, [{"proposal_id": "training-20", "verdict": "correct"}] * 61])
def test_a_malformed_or_oversized_batch_is_refused(tmp_path, synthetic_project, decisions):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28)])
    shots_before = len(project["shots"])
    with pytest.raises(ValueError):
        labeler.apply_action(project, "training", {"video_id": "training", "action": "validate_batch", "decisions": decisions}, root)
    assert len(project["shots"]) == shots_before and not project.get("reviews")


def test_load_review_marks_validatable_proposals_and_summarizes_the_inbox(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28), proposal(62, 70), proposal(40, 48, zone=None)])
    payload = review.load_review(project, "training", root)
    flags = {p["contact_frame"]: p["validatable"] for p in payload["queue"]}
    assert flags == {20: True, 40: False, 62: True}
    summary = payload["inbox"]
    assert summary["clips"] == [{"video_id": "training", "pending": 2, "order": 0}] and summary["current_order"] == 0
    assert summary["pending_total"] == 2 and summary["analyzed_clips"] == 1 and summary["total_clips"] == 2
    # Deciding a card removes it from the pending count; so does an existing label near the same contact.
    labeler.apply_action(project, "training", batch({"proposal_id": "training-62", "verdict": "wrong"}), root)
    assert inbox.summarize(project, root)["pending_total"] == 1


def test_the_clip_inbox_sidecar_is_written_reused_and_rebuilt_when_proposals_change(tmp_path):
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28)])
    first = inbox.clip_inbox(root, "training")
    sidecar = root / "training" / "inbox.json"
    assert sidecar.is_file() and [i["id"] for i in first["items"]] == ["training-20"]
    inbox._CACHE.clear()
    assert inbox.clip_inbox(root, "training") == first  # read back from the sidecar, not rebuilt
    write_analysis(root, [proposal(20, 28), proposal(62, 70)])
    rebuilt = inbox.clip_inbox(root, "training")
    assert [i["id"] for i in rebuilt["items"]] == ["training-20", "training-62"]
    assert inbox.clip_inbox(root, "nothing-here") is None
    (root / "broken").mkdir()
    (root / "broken" / "proposals.json").write_text("{not json", encoding="utf-8")
    assert inbox.clip_inbox(root, "broken") is None
