import json

import numpy as np
import pytest

from shot_predict.court import Court, ZONES
from shot_predict.dataset import split_matches
from shot_predict.storage import add_shot, read_project, write_project
from shot_predict.video import clip_indices


@pytest.fixture
def court():
    return Court([[0, 200], [120, 200], [120, 0], [0, 0]])


@pytest.mark.parametrize("hitter", ["near", "far"])
@pytest.mark.parametrize("zone", ZONES)
def test_zone_target_roundtrip(court, hitter, zone):
    assert court.zone(*court.normalized(*court.target(zone, hitter)), hitter) == zone


def test_receiver_side_convention(court):
    assert court.zone(0.1, 0.1, "near") == "deuce_deep"
    assert court.zone(0.9, 0.1, "near") == "ad_deep"
    assert court.zone(0.1, 0.9, "far") == "ad_deep"
    assert court.zone(0.9, 0.9, "far") == "deuce_deep"


@pytest.mark.parametrize("x,y,hitter", [
    (-0.1, 0.2, "near"), (0.4, 0.8, "near"),
    (0.4, 0.2, "far"), (0.4, 0.5, "far"), (np.nan, 0.2, "near"),
])
def test_invalid_bounces(court, x, y, hitter):
    with pytest.raises(ValueError):
        court.zone(x, y, hitter)


def test_invalid_calibration():
    with pytest.raises(ValueError):
        Court([[0, 0], [10, 10], [0, 10], [10, 0]])
    with pytest.raises(ValueError):
        Court([[0, 0]] * 4)


@pytest.mark.parametrize("lead", [0, 100, 200, 300])
def test_precontact_only(lead):
    indices = clip_indices(100, 30, 0.6, 8, lead)
    assert len(indices) == 8
    assert indices == sorted(indices)
    assert min(indices) >= 0
    assert max(indices) <= 100 - max(1, int(np.ceil(lead * 30 / 1000)))
    assert 100 not in indices


def test_insufficient_context_errors():
    with pytest.raises(ValueError, match="Not enough"):
        clip_indices(2, 30, 0.6, 8, 100)


def test_atomic_labels_and_duplicate_guard(tmp_path, court):
    path = tmp_path / "project.json"
    project = read_project(path)
    project["videos"]["v"] = {"frames": 200, "corners": court.corners}
    shot = add_shot(project, "v", 60, 90, "near", "groundstroke", court.target("ad_deep", "near"))
    assert shot["zone"] == "ad_deep"
    write_project(path, project)
    assert read_project(path) == project
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="already labeled"):
        add_shot(project, "v", 60, 90, "near", "serve", court.target("ad_deep", "near"))
    with pytest.raises(ValueError, match="Contact must precede"):
        add_shot(project, "v", 90, 60, "near", "serve", (10, 10))
    path.write_text(json.dumps({"version": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        read_project(path)


def test_match_split_and_unknown_matches():
    project = {
        "videos": {"a": {"match_id": "match1"}, "b": {"match_id": "match1"}, "c": {"match_id": "match2"}},
        "shots": [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}],
    }
    train, val, held = split_matches(project, 42, ["match2"])
    assert {s["video_id"] for s in train} == {"a", "b"}
    assert [s["video_id"] for s in val] == ["c"]
    assert held == ["match2"]
    with pytest.raises(ValueError, match="Unknown"):
        split_matches(project, 42, ["missing"])
    with pytest.raises(ValueError, match="Both training"):
        split_matches(project, 42, ["match1", "match2"])
    project["shots"] = [{"video_id": "a"}, {"video_id": "b"}]
    with pytest.raises(ValueError, match="TWO"):
        split_matches(project, 42)
