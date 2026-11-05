import json
import shutil
import subprocess
import os
from pathlib import Path

import pytest

from shot_predict.autopilot.camera import CameraTrack
from shot_predict.court import Court, ZONES

CORNERS = [[463.9, 840.9], [1451.8, 839.5], [1254.5, 321.1], [664.1, 319.7]]


def node_or_skip():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js 18+ is required for the frontend tests.")
    version = subprocess.run([node, "-p", "process.versions.node"], capture_output=True, text=True, check=True, timeout=5)
    if int(version.stdout.strip().split(".", 1)[0]) < 18:
        pytest.skip("Node.js 18+ is required for the frontend tests.")
    return node


def sample(frame, x, accepted=True, reference="main"):
    corners = [[cx + x, cy] for cx, cy in CORNERS]
    return {"frame": frame, "accepted": accepted, "reference": reference if accepted else None,
            "corners": corners if accepted else None, "reason": "ok", "inliers": 30, "line_score": 0.9}


def test_review_logic_matches_python_geometry_and_camera_interpolation(tmp_path):
    node = node_or_skip()
    court = Court(CORNERS)
    samples = [sample(0, 0), sample(30, 12), sample(60, 30), sample(90, 30, accepted=False), sample(120, 5),
               sample(150, 9), sample(300, 9), sample(330, 9, reference="low")]
    track = CameraTrack(samples, 59.94)
    frames = [0, 7, 15, 30, 45, 60, 61, 75, 90, 119, 120, 135, 150, 151, 200, 300, 330, 331, -3]
    expected = [{"frame": f, "corners": None if track.corners_at(f) is None else track.corners_at(f).tolist()} for f in frames]
    fixture = {
        "corners": CORNERS,
        "pixels": [{"x": x, "y": y, "pixel": list(court.pixel(x, y))} for x in (0, 0.2, 0.5, 0.9, 1) for y in (0, 0.3, 0.5, 0.77, 1)],
        "targets": [{"zone": z, "hitter": h, "target": list(court.target(z, h))} for z in ZONES for h in ("near", "far")],
        "camera": {"track": {"step_frames": track.step, "samples": [s for s in samples if s["accepted"]]}, "expected": expected},
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    test_file = Path(__file__).with_name("review_logic.test.cjs")
    result = subprocess.run(
        [node, "--test", str(test_file)], capture_output=True, text=True, timeout=60,
        env={**os.environ, "REVIEW_FIXTURE": str(path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pass 14" in result.stdout and "fail 0" in result.stdout and "skipped 0" in result.stdout, result.stdout


def test_frontend_ball_speed_limit_matches_the_trackers():
    import re

    from shot_predict.autopilot.ball import GATE_PIXELS_PER_SECOND

    source = Path(__file__).parents[1].joinpath("shot_predict", "labeler_frontend", "review_logic.js").read_text(encoding="utf-8")
    assert float(re.search(r"BALL_PIXELS_PER_SECOND = (\d+)", source).group(1)) == GATE_PIXELS_PER_SECOND
