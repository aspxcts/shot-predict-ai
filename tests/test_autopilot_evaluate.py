import json

import pytest

from shot_predict.autopilot.evaluate import (
    evaluate_project, evaluate_video, frames_for, load_proposals, pair_events, summarize,
)

FPS = 50.0  # 1 frame = 20 ms


def shot(contact, bounce, hitter="near", zone="ad_deep", kind="groundstroke"):
    return {"contact_frame": contact, "bounce_frame": bounce, "hitter": hitter, "zone": zone, "shot_type": kind}


def proposal(contact, hitter="near", zone="ad_deep", bounce=None, kind="groundstroke", confidence=0.9):
    return {"contact_frame": contact, "hitter": hitter, "zone": zone, "bounce_frame": bounce,
            "shot_type": kind, "confidence": confidence}


def test_pairing_is_one_to_one_and_prefers_the_closest():
    # Equal distance: the earlier true event wins deterministically, the second stays unmatched.
    assert pair_events([100, 110], [105], 10) == [(0, 0)]
    assert pair_events([100], [101, 103], 5) == [(0, 0)]
    assert pair_events([100], [200], 5) == []
    assert pair_events([], [1, 2], 5) == []


def test_frame_tolerance_conversion_is_fps_aware():
    assert frames_for(100, 25) == 2 and frames_for(100, 59.94) == 5 and frames_for(100, 50) == 5


def test_video_scoring_counts_hits_misses_and_extra_proposals():
    shots = [shot(100, 130, "near", "ad_deep"), shot(200, 240, "far", "deuce_short"), shot(400, 430, "near", "middle_deep")]
    proposals = [
        proposal(101, "near", "ad_deep", bounce=130),                       # 20 ms off, everything right
        proposal(207, "near", "deuce_short", bounce=250),                   # 140 ms off, wrong hitter, bounce 200 ms off
        proposal(300, "far", "ad_short"),                                   # no labeled shot here
    ]
    result = evaluate_video(shots, proposals, FPS, restrict_to_span=False)
    assert result["labeled"] == 3 and result["proposals_scored"] == 3 and result["paired"] == 2
    assert result["recall_matches"] == {"50": 1, "100": 1, "200": 2, "500": 2}
    assert result["unmatched_proposals"] == 1
    rows = {r["contact_frame"]: r for r in result["rows"] if r["status"] != "unmatched_proposal"}
    assert rows[100]["status"] == "matched" and rows[100]["hitter_correct"] and rows[100]["zone_correct"]
    assert rows[100]["contact_error_ms"] == pytest.approx(20) and rows[100]["bounce_error_ms"] == 0
    assert rows[200]["hitter_correct"] is False and rows[200]["bounce_error_ms"] == pytest.approx(200)
    assert rows[400]["status"] == "missed"


def test_span_restriction_ignores_proposals_outside_the_labeled_region():
    shots = [shot(500, 530)]
    proposals = [proposal(500), proposal(10), proposal(1500)]
    assert evaluate_video(shots, proposals, FPS, True)["proposals_scored"] == 1
    assert evaluate_video(shots, proposals, FPS, False)["proposals_scored"] == 3
    assert evaluate_video(shots, proposals, FPS, True)["proposals_total"] == 3


def test_summary_pools_clips_and_reports_gates():
    first = evaluate_video(
        [shot(100, 130), shot(200, 230)],
        [proposal(100, bounce=130), proposal(201, bounce=232)], FPS, False,
    )
    second = evaluate_video([shot(100, 130, "far", "ad_short")], [proposal(100, "far", "ad_short", bounce=131), proposal(300)], FPS, False)
    summary = summarize({"a": first, "b": second})
    assert summary["labeled_shots"] == 3 and summary["proposals_scored"] == 4
    assert summary["contact_recall"]["within_100ms"] == {"matched": 3, "of": 3, "rate": 1.0}
    assert summary["hitter_accuracy"]["rate"] == 1.0 and summary["zone_agreement_of_paired"]["rate"] == 1.0
    assert summary["bounce_within_tolerance"]["rate"] == 1.0
    assert summary["unmatched_proposals"] == 1 and summary["unmatched_per_paired_shot"] == pytest.approx(1 / 3)
    assert all(summary["gates"].values())


def test_missing_bounce_and_zone_count_as_wrong_and_empty_input_is_safe():
    result = evaluate_video([shot(100, 130)], [proposal(100, zone=None, bounce=None)], FPS, False)
    summary = summarize({"a": result})
    assert summary["zone_agreement_of_paired"]["rate"] == 0.0 and summary["bounce_within_tolerance"]["rate"] == 0.0
    assert summary["median_abs_bounce_error_ms"] is None
    empty = summarize({})
    assert empty["labeled_shots"] == 0 and empty["contact_recall"]["within_100ms"]["rate"] is None
    assert empty["gates"]["contact_recall_100ms_at_least_0.80"] is None


def test_project_evaluation_uses_each_clips_own_fps_and_skips_unlabeled_clips(tmp_path):
    project = {
        "videos": {"a": {"path": "x/a.mp4", "fps": 25.0}, "b": {"path": "x/b.mp4", "fps": 60.0}},
        "shots": [{"video_id": "a", **shot(100, 120)}],
    }
    report = evaluate_project(project, {"a": [proposal(102, bounce=121)], "b": [proposal(5)]}, restrict_to_span=False)
    assert list(report["per_clip"]) == ["a.mp4"]
    assert report["summary"]["contact_recall"]["within_100ms"]["matched"] == 1
    folder = tmp_path / "a"
    folder.mkdir()
    (folder / "proposals.json").write_text(json.dumps({"proposals": [proposal(1)], "range": [0, 500]}))
    assert load_proposals(tmp_path, ["a", "missing"]) == {"a": [proposal(1)]}
    assert load_proposals(tmp_path, ["a"], with_ranges=True) == ({"a": [proposal(1)]}, {"a": (0, 500)})


def test_labels_outside_the_analyzed_range_are_not_scored():
    project = {
        "videos": {"a": {"path": "x/a.mp4", "fps": 25.0}},
        "shots": [{"video_id": "a", **shot(100, 120)}, {"video_id": "a", **shot(9000, 9020)}],
    }
    report = evaluate_project(project, {"a": [proposal(100, bounce=120)]}, restrict_to_span=False, ranges={"a": (0, 500)})
    assert report["summary"]["labeled_shots"] == 1
    assert report["summary"]["contact_recall"]["within_100ms"]["matched"] == 1
