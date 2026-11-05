import cv2
import numpy as np
import pytest

from shot_predict.autopilot import ball
from shot_predict.autopilot.ball import ball_candidates, court_roi, expand_box, link_tracklets, scan_candidates

SIZE = (640, 360)
CORNERS = np.array([[120, 330], [520, 330], [440, 90], [200, 90]], np.float32)


def blank():
    return np.full((SIZE[1], SIZE[0], 3), (150, 110, 70), np.uint8)


def with_dot(x, y, radius=2, color=(190, 235, 200)):
    frame = blank()
    cv2.circle(frame, (round(x), round(y)), radius, color, -1)
    return frame


def full_roi():
    return np.full((SIZE[1], SIZE[0]), 255, np.uint8)


def test_three_frame_difference_finds_the_ball_where_it_is_not_where_it_was_or_will_be():
    frames = [with_dot(200 + 12 * i, 150 + 5 * i) for i in range(3)]
    found = ball_candidates(*frames, full_roi())
    assert len(found) == 1
    assert found[0][0] == pytest.approx(212, abs=1.5) and found[0][1] == pytest.approx(155, abs=1.5)


def test_static_objects_and_noise_are_not_candidates():
    still = with_dot(300, 200)
    assert ball_candidates(still, still, still, full_roi()) == []
    rng = np.random.default_rng(0)
    noisy = [np.clip(blank() + rng.integers(-3, 4, blank().shape), 0, 255).astype(np.uint8) for _ in range(3)]
    assert ball_candidates(*noisy, full_roi()) == []


def test_roi_blocked_boxes_and_size_limits_are_applied():
    frames = [with_dot(200 + 12 * i, 150) for i in range(3)]
    roi = full_roi()
    roi[:, :150] = 0
    assert len(ball_candidates(*frames, roi)) == 1
    roi[:, :] = 0
    assert ball_candidates(*frames, roi) == []
    assert ball_candidates(*frames, full_roi(), blocked=[[190, 140, 240, 165]]) == []
    big = [with_dot(200 + 12 * i, 150, radius=20) for i in range(3)]
    assert ball_candidates(*big, full_roi()) == []


def test_court_roi_covers_court_and_margins_but_not_far_outside_it():
    from shot_predict.court import Court

    court = Court(CORNERS.tolist())
    roi = court_roi(CORNERS, SIZE)

    def inside(x, y):
        px, py = court.pixel(x, y)
        return roi[min(max(py, 0), SIZE[1] - 1), min(max(px, 0), SIZE[0] - 1)] == 255

    assert inside(0.3, 0.3) and inside(0.3, 0.1) and inside(0.3, 0.9)
    assert inside(-0.3, 0.5) and inside(1.3, 0.5) and inside(0.5, -0.4)
    assert not inside(-0.9, 0.5) and not inside(1.9, 0.5)
    assert roi[5, 5] == 0


def test_signage_and_logo_hot_spots_are_removed_but_a_moving_ball_is_kept():
    ball_path = {f: [(50.0 + 9 * f, 200.0 - 2 * f, 10, 40.0)] for f in range(60)}
    candidates = {f: list(items) + [(300.0, 120.0, 8, 25.0)] for f, items in ball_path.items()}
    cleaned = ball.drop_persistent_noise(candidates, SIZE)
    assert all(len(items) == 1 for items in cleaned.values())
    assert [items[0][:2] for items in cleaned.values()] == [items[0][:2] for items in ball_path.values()]
    # A short hover (serve toss apex) is below the persistence threshold and survives.
    brief = {f: [(300.0, 120.0, 8, 25.0)] if f < 8 else [] for f in range(60)}
    assert sum(len(v) for v in ball.drop_persistent_noise(brief, SIZE).values()) == 8


def test_expand_box_grows_by_a_fraction_of_height_with_a_minimum():
    assert expand_box(None) is None
    assert expand_box([100, 100, 140, 200], fraction=0.1) == [90, 90, 150, 210]
    assert expand_box([100, 100, 110, 110], fraction=0.1, minimum=8) == [92, 92, 118, 118]


def test_tracklet_keeps_gaps_and_never_interpolates():
    candidates = {f: [(100 + 14 * f, 200 - 6 * f, 12, 40.0)] for f in range(0, 12) if f not in (4, 5)}
    (track,) = link_tracklets(candidates, fps=60)
    assert track["frames"] == [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]
    assert len(track["xy"]) == 10 and track["xy"][4] == [184.0, 164.0]


def test_tracklet_survives_a_bounce_like_velocity_change_and_splits_on_a_long_gap():
    path = [(100 + 15 * f, 100 + 12 * f) for f in range(8)] + [(100 + 15 * f, 100 + 12 * 7 - 8 * (f - 7)) for f in range(8, 14)]
    candidates = {f: [(x, y, 10, 30.0)] for f, (x, y) in enumerate(path)}
    (track,) = link_tracklets(candidates, fps=60)
    assert track["frames"] == list(range(14))
    later = {f + 30: [(x, y, 10, 30.0)] for f, (x, y) in enumerate(path)}
    tracks = link_tracklets({**candidates, **later}, fps=60)
    assert len(tracks) == 2 and tracks[0]["frames"][-1] == 13 and tracks[1]["frames"][0] == 30


def test_clutter_and_stationary_points_do_not_make_tracklets():
    rng = np.random.default_rng(1)
    clutter = {f: [(float(rng.uniform(0, 600)), float(rng.uniform(0, 300)), 9, 20.0) for _ in range(3)] for f in range(40)}
    assert link_tracklets(clutter, fps=60) == []
    stationary = {f: [(300.0 + rng.normal(0, 0.5), 200.0 + rng.normal(0, 0.5), 9, 20.0)] for f in range(20)}
    assert link_tracklets(stationary, fps=60) == []
    assert link_tracklets({}, fps=60) == []


def test_ball_follows_the_true_object_among_decoys():
    candidates = {}
    for f in range(12):
        candidates[f] = [(100 + 14 * f, 150 + 3 * f, 12, 40.0), (500.0, 40.0 + (f % 2), 8, 20.0)]
    tracklets = link_tracklets(candidates, fps=60)
    assert len(tracklets) == 1 and tracklets[0]["xy"][0] == [100.0, 150.0] and tracklets[0]["frames"][-1] == 11


def test_flat_line_shimmer_is_not_a_flight_but_a_gentle_real_arc_is():
    shimmer = {f: [(100.0 + 40 * f, 833.0 + (f % 2), 9, 20.0)] for f in range(10)}
    assert link_tracklets(shimmer, fps=60) == []
    arc = {f: [(100.0 + 30 * f, 400.0 - 3 * f + 0.9 * f * f, 9, 20.0)] for f in range(12)}
    assert len(link_tracklets(arc, fps=60)) == 1


def test_gate_scales_with_frame_rate():
    slow = {f: [(100 + 100 * f, 150.0 + 12 * f, 10, 30.0)] for f in range(8)}
    assert len(link_tracklets(slow, fps=25)) == 1
    assert link_tracklets(slow, fps=120) == []


def test_scan_candidates_reads_sequentially_and_respects_roi_and_boxes(tmp_path):
    path = tmp_path / "ball.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, SIZE)
    for i in range(14):
        writer.write(with_dot(60 + 20 * i, 200 - 4 * i, radius=3))
    writer.release()
    found = scan_candidates(path, 2, 12, lambda f: full_roi(), lambda f: None)
    assert sorted(found) == list(range(2, 12))
    for frame, items in found.items():
        assert len(items) == 1 and items[0][0] == pytest.approx(60 + 20 * frame, abs=2.5)
    skipped = scan_candidates(path, 2, 12, lambda f: None if f % 2 else full_roi(), lambda f: None)
    assert sorted(skipped) == [2, 4, 6, 8, 10]
    (track,) = link_tracklets(found, fps=30)
    assert track["frames"][0] == 2 and track["frames"][-1] == 11


def test_court_roi_keeps_lines_and_the_net_and_excludes_far_outside_regions():
    from shot_predict.court import Court

    corners = [[460, 840], [1450, 840], [1255, 320], [660, 320]]
    court = Court(corners)
    roi = court_roi(np.array(corners, np.float32), (2400, 2000))

    def inside(x, y):
        px, py = court.pixel(x, y)
        return roi[py, px] == 255

    # Lines and the net are searched too: a ball passing over them must stay detectable.
    for x, y in [(0.3, 0.3), (0.7, 0.62), (0.5, 0.1), (0.5, 0.0), (0.5, 1.0), (0.0, 0.4), (0.5, 0.5), (0.5, 0.2308)]:
        assert inside(x, y), (x, y)
    assert inside(-0.3, 0.5) and inside(0.5, -0.4) and inside(0.5, 1.1)  # margins around the court
    assert not inside(-0.45, 0.5) and not inside(0.5, 1.25)


def test_gap_tolerance_is_a_duration_so_sixty_fps_footage_bridges_more_frames():
    def broken_flight(fps, gap):
        # `gap` is the frame difference across the break (a detection every frame has gap 1).
        frames = list(range(0, 8)) + list(range(7 + gap, 15 + gap))
        return {f: [(100.0 + 20 * f * 25 / fps, 150.0 + 4 * f * 25 / fps, 20, 1.0)] for f in frames}

    # 0.12 s: 3 frames at 25 fps, 7 at 60 fps. One frame more than that splits the flight.
    assert len(link_tracklets(broken_flight(25, 3), fps=25)) == 1
    assert len(link_tracklets(broken_flight(25, 4), fps=25)) == 2
    assert len(link_tracklets(broken_flight(60, 7), fps=60)) == 1
    assert len(link_tracklets(broken_flight(60, 8), fps=60)) == 2


def test_noisy_frames_stay_fast_because_strengths_are_computed_in_one_pass():
    import time

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (SIZE[1], SIZE[0]), dtype=np.uint8)
    frames = [np.clip(base.astype(int) + rng.integers(-40, 41, base.shape), 0, 255).astype(np.uint8) for _ in range(3)]
    start = time.perf_counter()
    for _ in range(5):
        ball.ball_candidates_gray(*frames, full_roi())
    assert (time.perf_counter() - start) / 5 < 0.25

