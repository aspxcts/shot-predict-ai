import json

import pytest

from shot_predict import labeler
from shot_predict.autopilot import review
from shot_predict.court import Court
from shot_predict.storage import read_project, write_project

# The synthetic clips are 80 frames at 20 fps; labeling needs 18+ frames of pre-contact context.
CORNERS = [[20, 110], [140, 110], [140, 10], [20, 10]]
COURT = Court(CORNERS)


def proposal(contact, bounce=None, hitter="near", zone="ad_deep", confidence=0.8, bounce_confidence=0.4, **extra):
    return {
        "id": f"training-{contact}", "contact_frame": contact, "hitter": hitter, "hitter_confidence": confidence,
        "shot_type": "groundstroke", "confidence": confidence,
        "bounce_frame": bounce, "bounce_confidence": bounce_confidence,
        "landing_pixel": list(COURT.target(zone, hitter)) if zone else None, "zone": zone,
        "contact_corners": CORNERS, "bounce_corners": CORNERS,
        "players": {"near": [50, 70, 70, 108], "far": None}, "ball_trail": [[contact, 60.0, 80.0], [contact + 1, 61.0, 78.0]],
        "reasons": ["flights_meet_3px_gap_2f"], **extra,
    }


def write_analysis(root, proposals, camera=None, video_id="training"):
    folder = root / video_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "proposals.json").write_text(json.dumps({"version": "t", "proposals": proposals}), encoding="utf-8")
    if camera is not None:
        (folder / "camera.json").write_text(json.dumps(camera), encoding="utf-8")


def camera_track(shift_per_sample=0.0, accepted_until=70):
    samples = []
    for frame in range(0, 80, 10):
        ok = frame <= accepted_until
        corners = [[x + shift_per_sample * frame / 10, y] for x, y in CORNERS]
        samples.append({"frame": frame, "accepted": ok, "reference": "main" if ok else None,
                        "corners": corners if ok else None, "reason": "ok", "inliers": 50, "line_score": 0.9})
    return {"fps": 20.0, "step_frames": 10, "samples": samples}


def event(action, **fields):
    return {"video_id": "training", "action": action, **fields}


def accept_fields(**changes):
    return {"proposal_id": "training-20", "contact_frame": 20, "bounce_frame": 28, "hitter": "near",
            "shot_type": "groundstroke", "landing_pixel": list(COURT.target("ad_deep", "near")), **changes}


def test_queue_hides_saved_and_reviewed_proposals_and_reports_stats(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    # The synthetic project already has labels at frames 30 and 50; 31 is the same shot within 0.25 s.
    write_analysis(root, [proposal(20, 28), proposal(31, 40), proposal(62, 70), proposal(22, 29, id="training-dup"),
                          proposal(20, 28)])
    queue = review.load_review(project, "training", root)
    assert [p["contact_frame"] for p in queue["queue"]] == [20, 22, 62]
    assert queue["dropped"] == 0 and queue["camera"] is None and queue["stats"]["proposals"] == 4
    review.reject_proposal(project, "training", {"proposal_id": "training-62", "decision": "rejected"}, root)
    again = review.load_review(project, "training", root)
    assert [p["contact_frame"] for p in again["queue"]] == [20, 22]
    assert again["stats"]["rejected"] == 1


def test_malformed_and_unreadable_analysis_never_crashes_the_queue(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    assert "No analysis" in review.load_review(project, "training", root)["message"]
    good = proposal(20, 28)
    bad = [
        {**proposal(11, 21), "hitter": "left"}, {**proposal(12, 22), "id": ""}, proposal(13, 13), proposal(14, 90),
        {**proposal(15, 25), "confidence": float("nan")}, {**proposal(16, 26), "confidence": 1.5},
        {**proposal(17, 27), "zone": "nowhere"}, {**proposal(18, 28), "bounce_corners": [[1, 2]]}, "text", None,
        {**proposal(19, 29), "shot_type": "lob"}, proposal(200, None),
    ]
    write_analysis(root, [good, *bad])
    loaded = review.load_review(project, "training", root)
    assert [p["contact_frame"] for p in loaded["queue"]] == [20] and loaded["dropped"] == len(bad)
    (root / "training" / "proposals.json").write_text("{not json", encoding="utf-8")
    assert "unreadable" in review.load_review(project, "training", root)["message"]
    (root / "training" / "proposals.json").write_text(json.dumps({"proposals": "oops"}), encoding="utf-8")
    assert review.load_review(project, "training", root)["queue"] == []


def test_accepting_an_unchanged_proposal_records_auto_confirmed_provenance(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28)])
    before = len(project["shots"])
    result = labeler.apply_action(project, "training", event("accept_proposal", review_seconds=3.5, **accept_fields()), root)
    assert len(project["shots"]) == before + 1 and result["edited_fields"] == []
    shot = project["shots"][-1]
    assert shot["source"] == "auto_confirmed" and shot["zone"] == "ad_deep" and shot["review_seconds"] == 3.5
    assert shot["proposal"]["id"] == "training-20" and "ball_trail" not in shot["proposal"]
    assert "input_court_corners" not in shot and shot["court_corners"] == project["videos"]["training"]["corners"]
    assert review.load_review(project, "training", root)["queue"] == []
    assert review.load_review(project, "training", root)["stats"]["auto_confirmed"] == 1
    labeler.apply_action(project, "training", event("undo", shot_id=shot["id"]), root)
    assert [p["id"] for p in review.load_review(project, "training", root)["queue"]] == ["training-20"]


def test_any_human_change_is_recorded_as_auto_corrected_with_the_edited_fields(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28)])
    labeler.apply_action(project, "training", event("accept_proposal", **accept_fields(
        contact_frame=21, bounce_frame=29, hitter="far", shot_type="serve",
        landing_pixel=list(COURT.target("deuce_short", "far")),
    )), root)
    shot = project["shots"][-1]
    assert shot["source"] == "auto_corrected" and shot["zone"] == "deuce_short" and shot["hitter"] == "far"
    assert set(shot["edited_fields"]) == {"contact_frame", "bounce_frame", "hitter", "shot_type", "landing"}
    assert shot["proposal"]["contact_frame"] == 20 and shot["proposal"]["zone"] == "ad_deep"


def test_a_clicked_landing_on_a_proposal_without_one_counts_as_an_edit(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28, zone=None, bounce_confidence=0.0)])
    labeler.apply_action(project, "training", event("accept_proposal", **accept_fields(
        landing_pixel=list(COURT.target("middle_deep", "near")))), root)
    assert project["shots"][-1]["edited_fields"] == ["landing"] and project["shots"][-1]["zone"] == "middle_deep"


def test_acceptance_is_validated_against_the_saved_proposal_not_the_browser(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28)])
    for change, message in [
        ({"proposal_id": "training-999"}, "no longer available"), ({"landing_pixel": [9999, 5]}, "inside the video"),
        ({"landing_pixel": [True, 5]}, "finite"), ({"contact_frame": 20.5}, "whole frame"),
        ({"bounce_frame": 20}, "precede"), ({"review_seconds": -1}, "invalid"), ({"hitter": "left"}, "near or far"),
        ({"landing_pixel": list(COURT.target("ad_deep", "far"))}, "opposite side"),
    ]:
        with pytest.raises(ValueError, match=message):
            labeler.apply_action(project, "training", event("accept_proposal", **accept_fields(**change)), root)
    assert len(project["shots"]) == 4


def test_moving_camera_clips_resolve_corners_from_the_track_at_each_frame(tmp_path, synthetic_project):
    _, project = synthetic_project
    project["videos"]["training"]["corners"] = None
    project["shots"] = [s for s in project["shots"] if s["video_id"] != "training"]
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 40)], camera=camera_track(shift_per_sample=2.0))
    landing = list(Court([[x + 2.0 * 4, y] for x, y in CORNERS]).target("ad_deep", "near"))  # frame 40 = sample 4
    labeler.apply_action(project, "training", event("accept_proposal", **accept_fields(
        bounce_frame=40, landing_pixel=landing)), root)
    shot = project["shots"][-1]
    assert shot["zone"] == "ad_deep" and shot["edited_fields"] == ["landing"]  # same frames, but a different clicked pixel
    assert shot["court_corners"][0][0] == pytest.approx(20 + 2.0 * 4)
    assert shot["input_court_corners"][0][0] == pytest.approx(20 + 2.0 * 2)  # contact frame 20 = sample 2
    payload = review.load_review(project, "training", root)
    assert payload["camera"]["step_frames"] == 10 and len(payload["camera"]["samples"]) == 8


def test_frames_outside_a_court_view_cannot_be_saved(tmp_path, synthetic_project):
    _, project = synthetic_project
    project["videos"]["training"]["corners"] = None
    project["shots"] = []
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 75)], camera=camera_track(accepted_until=40))
    with pytest.raises(ValueError, match="not in a recognized court-view"):
        labeler.apply_action(project, "training", event("accept_proposal", **accept_fields(bounce_frame=75)), root)
    with pytest.raises(ValueError, match="not in a recognized court-view"):
        labeler.apply_action(project, "training", event(
            "save_shot", contact_frame=20, bounce_frame=75, hitter="near", shot_type="groundstroke",
            landing_pixel=list(COURT.target("ad_deep", "near")),
        ), root)
    assert project["shots"] == []


def test_manual_save_on_a_moving_camera_clip_uses_the_track_and_fixed_clips_are_unchanged(tmp_path, synthetic_project):
    _, project = synthetic_project
    root = tmp_path / "analysis"
    project["videos"]["training"]["corners"] = None
    project["shots"] = []
    write_analysis(root, [], camera=camera_track())
    labeler.apply_action(project, "training", event(
        "save_shot", contact_frame=20, bounce_frame=30, hitter="near", shot_type="groundstroke",
        landing_pixel=list(COURT.target("deuce_deep", "near")),
    ), root)
    manual = project["shots"][-1]
    assert manual["zone"] == "deuce_deep" and "source" not in manual and manual["input_court_corners"] == CORNERS
    fixed_root = tmp_path / "none"
    labeler.apply_action(project, "validation", {
        "video_id": "validation", "action": "save_shot", "contact_frame": 22, "bounce_frame": 35, "hitter": "far",
        "shot_type": "groundstroke", "landing_pixel": list(COURT.target("ad_short", "far")),
    }, fixed_root)
    assert "input_court_corners" not in project["shots"][-1]
    with pytest.raises(ValueError, match="Calibrate"):
        labeler.apply_action(project, "training", event(
            "save_shot", contact_frame=40, bounce_frame=50, hitter="near", shot_type="groundstroke", landing_pixel=[50, 50],
        ), fixed_root)


def test_reject_out_skip_and_undo_review(tmp_path, synthetic_project):
    path, project = synthetic_project
    root = tmp_path / "analysis"
    write_analysis(root, [proposal(20, 28), proposal(62, 70)])
    for pid, decision in [("training-20", "out"), ("training-62", "skipped")]:
        labeler.apply_action(project, "training", event("reject_proposal", proposal_id=pid, decision=decision), root)
    assert review.load_review(project, "training", root)["queue"] == []
    labeler.apply_action(project, "training", event("undo_review", proposal_id="training-62"), root)
    assert [p["id"] for p in review.load_review(project, "training", root)["queue"]] == ["training-62"]
    with pytest.raises(ValueError, match="no longer exists"):
        labeler.apply_action(project, "training", event("undo_review", proposal_id="training-62"), root)
    with pytest.raises(ValueError):
        labeler.apply_action(project, "training", event("reject_proposal", proposal_id="training-62", decision="maybe"), root)
    write_project(path, project)
    assert {r["decision"] for r in read_project(path)["reviews"]} == {"out"}
