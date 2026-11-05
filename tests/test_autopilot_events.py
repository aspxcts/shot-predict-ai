import numpy as np
import pytest

from shot_predict.autopilot import ball, events
from shot_predict.court import Court

FPS = 60.0
CORNERS = [[330, 640], [950, 640], [820, 250], [460, 250]]
COURT = Court(CORNERS)
BOXES = {"near": [590, 520, 690, 650, 0.9], "far": [520, 205, 600, 270, 0.9]}


def boxes_at(frame, side):
    return BOXES[side]


def corners_at(frame):
    return np.array(CORNERS, float)


def lerp(a, b, u):
    return (1 - u) * np.asarray(a, float) + u * np.asarray(b, float)


def rally():
    """near hits at 100 -> bounce 130; far hits at 160 -> bounce 195; near hits at 225 -> bounce 255; far at 285."""
    contacts = [  # frame, hitter, racket ground point, racket height (px)
        (100, "near", (640, 640), 40), (160, "far", (560, 260), 22), (225, "near", (610, 640), 40), (285, "far", (580, 260), 22),
    ]
    landings = [(130, "deuce_deep", "near"), (195, "ad_short", "far"), (255, "middle_deep", "near")]
    truth = []
    path = {}
    for i, (frame, hitter, ground, racket) in enumerate(contacts[:-1]):
        bounce_frame, zone, side = landings[i]
        target = np.asarray(COURT.target(zone, side), float)
        next_frame, _, next_ground, next_racket = contacts[i + 1]
        for t in range(frame, bounce_frame):
            u = (t - frame) / (bounce_frame - frame)
            path[t] = lerp(ground, target, u) - [0, racket * (1 - u) + 90 * 4 * u * (1 - u)]
        for t in range(bounce_frame, next_frame):
            u = (t - bounce_frame) / (next_frame - bounce_frame)
            path[t] = lerp(target, next_ground, u) - [0, next_racket * u + 55 * 4 * u * (1 - u)]
        truth.append((frame, hitter, bounce_frame, zone))
    return path, truth


def candidates_from(path, drop_rng=None, near_contacts=(100, 160, 225, 285), near_bounces=(130, 195, 255)):
    rng = np.random.default_rng(3)
    found = {}
    for t, (x, y) in path.items():
        if any(abs(t - c) <= 3 for c in near_contacts) or any(abs(t - b) <= 1 for b in near_bounces):
            continue
        if drop_rng and rng.random() < drop_rng:
            continue
        found[t] = [(float(x), float(y), 10, 40.0)]
    return found


def test_rally_contacts_hitters_bounces_and_zones_are_recovered_with_realistic_gaps():
    path, truth = rally()
    tracks = ball.link_tracklets(candidates_from(path, drop_rng=0.15), FPS)
    proposals = events.propose_shots(tracks, FPS, corners_at, boxes_at, (90, 300), "vid")
    assert len(proposals) >= 3
    matched = 0
    for frame, hitter, bounce, zone in truth:
        # The synthetic ball is tracked without delay, so a far hitter's corrected contact lands one lead early.
        lead = round(events.FAR_CONTACT_LEAD_SECONDS * FPS) if hitter == "far" else 0
        near = [p for p in proposals if abs(p["contact_frame"] - (frame - lead)) <= 6]
        if not near:
            continue
        p = near[0]
        matched += 1
        assert p["hitter"] == hitter
        if p["bounce_frame"] is not None:
            assert abs(p["bounce_frame"] - bounce) <= 5
            assert p["zone"] == zone
            assert p["bounce_corners"] is not None and p["landing_pixel"] is not None
    assert matched >= 3
    assert all(0 <= p["confidence"] <= 1 and 0 <= p["bounce_confidence"] <= events.BOUNCE_CONFIDENCE_CAP for p in proposals)
    assert len({p["id"] for p in proposals}) == len(proposals)
    lead = round(events.FAR_CONTACT_LEAD_SECONDS * FPS)
    for p in proposals:
        engine_frame = int(p["id"].split("-")[1])
        assert engine_frame - p["contact_frame"] == (lead if p["hitter"] == "far" else 0), p["id"]


def test_balls_that_never_meet_a_player_are_penalised_and_not_confident():
    path, _ = rally()
    far_away = lambda frame, side: [5, 5, 40, 60, 0.9]
    tracks = ball.link_tracklets(candidates_from(path), FPS)
    proposals = events.propose_shots(tracks, FPS, corners_at, far_away, (90, 300), "vid")
    assert proposals and all(p["confidence"] < 0.5 for p in proposals)
    assert all("ball_not_at_hitter" in p["reasons"] for p in proposals)
    at_players = events.propose_shots(tracks, FPS, corners_at, boxes_at, (90, 300), "vid")
    assert max(p["confidence"] for p in at_players) > 0.8


def test_bounce_is_the_largest_downward_kink_and_straight_flights_have_none():
    kink = [(f, 400.0 + 3 * f, 200.0 + 9 * f) for f in range(0, 12)]
    kink += [(f, 400.0 + 3 * f, 200.0 + 9 * 11 - 4 * (f - 11)) for f in range(12, 24)]
    frame, x, y, confidence, drop = events.find_bounce(kink, FPS)
    assert abs(frame - 11) <= 1 and y == pytest.approx(200 + 99, abs=8) and drop == pytest.approx(13, abs=2.5) and confidence > 0.3
    straight = [(f, 400.0 + 3 * f, 200.0 + 9 * f) for f in range(24)]
    assert events.find_bounce(straight, FPS) is None
    assert events.find_bounce(straight[:5], FPS) is None
    gap = [p for p in kink if not 9 <= p[0] <= 13]
    assert abs(events.find_bounce(gap, FPS)[0] - 11) <= 2


def test_reversals_split_a_single_tracklet_but_leave_smooth_flights_alone():
    down = [(f, 300.0 + 4 * f, 200.0 + 8 * f) for f in range(10)]
    up = [(f, 300.0 + 4 * f, 200.0 + 8 * 9 - 8 * (f - 9)) for f in range(10, 20)]
    one = {"frames": [p[0] for p in down + up], "xy": [[p[1], p[2]] for p in down + up]}
    pieces = events.split_at_reversals([one], FPS)
    assert len(pieces) == 2 and abs(pieces[0]["frames"][-1] - 9) <= 1
    smooth = {"frames": list(range(20)), "xy": [[300.0 + 4 * f, 200.0 + 8 * f] for f in range(20)]}
    assert events.split_at_reversals([smooth], FPS) == [smooth]


def test_first_flight_in_a_segment_that_starts_at_the_hitter_is_a_serve():
    frames = list(range(100, 124))
    xy = [[640 - 2 * (f - 100), 630.0 - 9 * (f - 100)] for f in frames]
    tracks = [{"frames": frames, "xy": xy}]
    proposals = events.propose_shots(tracks, FPS, corners_at, boxes_at, (90, 200), "vid")
    assert len(proposals) == 1
    p = proposals[0]
    assert p["hitter"] == "near" and p["shot_type"] == "serve" and p["confidence"] >= 0.5
    assert any(r.startswith("flight_starts_at_hitter") for r in p["reasons"])
    # No kink: only a hint where the visible flight ends, never a landing or zone guess.
    assert p["bounce_frame"] == 124 and p["bounce_confidence"] == 0.0
    assert p["landing_pixel"] is None and p["zone"] is None and "bounce_hint_from_end_of_visible_flight" in p["reasons"]


def test_overlapping_tracklets_are_skipped_instead_of_crashing():
    a = {"frames": list(range(100, 130)), "xy": [[300.0 + 3 * f, 500.0 - 6 * f] for f in range(30)]}
    b = {"frames": list(range(105, 135)), "xy": [[400.0 + 3 * f, 200.0 + 6 * f] for f in range(30)]}
    assert events._meeting(a, b) is None
    assert events.find_contacts([a, b], FPS, boxes_at, 90) is not None


def test_landing_outside_the_court_is_flagged_instead_of_assigned_a_zone():
    kink = [(f, 640.0 + 40 * f, 300.0 + 9 * f) for f in range(0, 12)]
    kink += [(f, 640.0 + 40 * f, 300.0 + 9 * 11 - 4 * (f - 11)) for f in range(12, 24)]
    zone, normalized = events._landing((1100.0, 400.0), CORNERS, "far")
    assert zone is None and (normalized[0] > 1 or normalized[0] < 0 or normalized[1] < 0.5)


def test_follow_flight_crosses_a_bounce_but_ignores_unrelated_pieces():
    down = {"frames": list(range(0, 12)), "xy": [[400.0 + 3 * f, 200.0 + 9 * f] for f in range(12)]}
    up = {"frames": list(range(14, 26)), "xy": [[400.0 + 3 * f, 299.0 - 4 * (f - 11)] for f in range(14, 26)]}
    noise_in_time = {"frames": list(range(13, 26)), "xy": [[50.0 + 5 * f, 600.0] for f in range(13, 26)]}
    too_late = {"frames": list(range(60, 70)), "xy": [[400.0 + 3 * f, 299.0 - 4 * (f - 11)] for f in range(60, 70)]}
    pieces = [down, noise_in_time, up, too_late]
    assert events.follow_flight(pieces, 0, set(), 200, FPS) == [0, 2]
    assert events.follow_flight(pieces, 0, {2}, 200, FPS) == [0]  # the next shot's own flight is off limits
    assert events.follow_flight(pieces, 0, set(), 20, FPS) == [0, 2]
    assert events.follow_flight(pieces, 0, set(), 5, FPS) == [0]


def test_a_proposals_trail_holds_only_its_own_flights_even_when_noise_shares_the_time_window():
    path, _ = rally()
    tracks = ball.link_tracklets(candidates_from(path), FPS)
    # A zig-zagging tracklet far from the ball for the whole rally: a tangle if it ever joined a trail.
    noise = {"frames": list(range(95, 290)), "xy": [[120.0 + 0.5 * (f - 95), 480.0 + 60.0 * ((f % 4) - 1.5)] for f in range(95, 290)]}
    proposals = events.propose_shots([*tracks, noise], FPS, corners_at, boxes_at, (90, 300), "vid")
    assert len(proposals) >= 3
    noise_points = {(f, round(x, 1), round(y, 1)) for f, (x, y) in zip(noise["frames"], noise["xy"])}
    gate = ball.GATE_PIXELS_PER_SECOND / FPS
    for p in proposals:
        trail = p["ball_trail"]
        assert trail and not any(tuple(point) in noise_points for point in trail)
        for a, b in zip(trail, trail[1:]):
            assert np.hypot(b[1] - a[1], b[2] - a[2]) <= gate * max(1, b[0] - a[0])
    assert any(p["bounce_frame"] is not None and p["zone"] for p in proposals)
