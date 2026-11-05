import json

import pytest

from shot_predict.autopilot import ball, events, pipeline, playerpass

CORNERS = [[330, 640], [950, 640], [820, 250], [460, 250]]


@pytest.fixture
def clip(tmp_path):
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"not a real video: every heavy stage is stubbed")
    video = {"path": str(video_file), "fps": 30.0, "frames": 300, "width": 1280, "height": 720, "corners": CORNERS}
    return {"videos": {"vid": video}, "shots": []}, tmp_path / "analysis"


@pytest.fixture
def stages(monkeypatch):
    calls = {"players": 0, "ball": 0, "proposals": 0, "strides": []}

    def detect(path, ranges, corners_at, cropper, stride=3, progress=None):
        calls["players"] += 1
        calls["strides"].append(stride)
        return {f: {"near": [590, 520, 690, 650, 0.9], "far": [520, 205, 600, 270, 0.9]} for start, end in ranges for f in range(start, end, 3)}

    def scan(path, start, end, roi_for_frame, boxes_for_frame, progress=None):
        calls["ball"] += 1
        return {}

    real_propose = events.propose_shots

    def propose(*args, **kwargs):
        calls["proposals"] += 1
        return real_propose(*args, **kwargs)

    monkeypatch.setattr(playerpass, "detect_player_samples", detect)
    monkeypatch.setattr(ball, "scan_candidates", scan)
    monkeypatch.setattr(events, "propose_shots", propose)
    return calls


def test_a_second_analysis_reuses_the_player_and_ball_passes_and_only_redoes_proposals(clip, stages):
    project, root = clip
    first = pipeline.analyze_clip(project, "vid", root, [], cropper=None, log=lambda message: None)
    assert first["segments"] and all(isinstance(segment, list) for segment in first["segments"])
    assert stages["players"] == 1 and stages["ball"] == len(first["segments"]) and stages["proposals"] == len(first["segments"])

    again = pipeline.analyze_clip(project, "vid", root, [], cropper=None, log=lambda message: None)
    assert stages["players"] == 1, "player detection must not rerun when its sidecar is valid"
    assert stages["ball"] == len(first["segments"]), "the ball scan must not rerun when its sidecar is valid"
    assert stages["proposals"] == 2 * len(first["segments"]), "proposals are cheap and always rebuilt from the sidecars"
    assert again["segments"] == first["segments"]
    saved = json.loads((root / "vid" / "proposals.json").read_text(encoding="utf-8"))
    assert saved["segments"] == first["segments"]


def test_changed_segments_invalidate_the_cached_passes(clip, stages):
    project, root = clip
    pipeline.analyze_clip(project, "vid", root, [], cropper=None, log=lambda message: None)
    ball_path = root / "vid" / "ball.json"
    cached = json.loads(ball_path.read_text(encoding="utf-8"))
    cached["segments"] = [[0, 10]]
    ball_path.write_text(json.dumps(cached), encoding="utf-8")
    pipeline.analyze_clip(project, "vid", root, [], cropper=None, log=lambda message: None)
    assert stages["ball"] == 2 and stages["players"] == 1


def test_player_detection_runs_at_a_fixed_time_step_and_only_a_coarser_cache_is_redone(clip, stages):
    project, root = clip  # 30 fps: 0.12 s is every 4th frame (25 fps Turin footage: every 3rd; 60 fps: every 7th)
    assert [playerpass.stride_for(fps) for fps in (25, 30, 50, 59.94)] == [3, 4, 6, 7]
    assert playerpass.stride_for(1) == 1
    pipeline.analyze_clip(project, "vid", root, [], cropper=None, log=lambda message: None)
    assert stages["strides"] == [4]
    players_path = root / "vid" / "players.json"

    def rerun_with_cached_stride(value):
        cached = json.loads(players_path.read_text(encoding="utf-8"))
        if value is None:
            cached.pop("stride")
        else:
            cached["stride"] = value
        players_path.write_text(json.dumps(cached), encoding="utf-8")
        pipeline.analyze_clip(project, "vid", root, [], cropper=None, log=lambda message: None)

    rerun_with_cached_stride(2)  # sampled more finely than needed: reuse
    rerun_with_cached_stride(None)  # an older cache without a stride was taken every 3rd frame: reuse
    assert stages["players"] == 1
    rerun_with_cached_stride(9)  # coarser than needed: detect again at the requested step
    assert stages["players"] == 2 and stages["strides"] == [4, 4]
    assert json.loads(players_path.read_text(encoding="utf-8"))["stride"] == 4
