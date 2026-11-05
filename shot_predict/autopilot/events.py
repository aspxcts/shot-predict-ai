"""Turn ball tracklets into shot proposals: contact, hitter, bounce, landing zone.

Physical cues used (all measured on image coordinates of a steady camera):
* Contact: an incoming flight and an outgoing flight meet at a player and their vertical image velocity
  flips sign. The hitter is the end the outgoing ball travels away from (up the image = near hitter).
* Bounce: at the ground the ball's vertical image velocity drops sharply (its height velocity reverses).
  The landing point is the ball's position at that frame, mapped through the court homography.
Everything is a proposal with a confidence and reasons; a human confirms it.
"""

import numpy as np

from ..court import Court, zone_for

ENGINE_VERSION = "events-2"  # events-2: far-hitter contacts moved 0.12 s earlier
FIT_POINTS = 6
MIN_TRACK_POINTS = 8
MIN_SPEED_PX_PER_SECOND = 150.0
MIN_NET_DISPLACEMENT = 30.0
MAX_CONTACT_GAP_SECONDS = 0.8
MAX_MEETING_DISTANCE = 60.0
MAX_BOUNCE_SEARCH_SECONDS = 2.6
MIN_BOUNCE_DROP_PX_PER_SECOND = 150.0
DUPLICATE_CONTACT_SECONDS = 0.4
HITTER_BOX_GROWTH = 0.45
HITTER_FLIGHT_MIN_POINTS = 8
HITTER_FLIGHT_MIN_NET = 80.0
HITTER_FLIGHT_REACH = 1.2
BOUNCE_OVERRUN_SECONDS = 0.2
BOUNCE_CONFIDENCE_CAP = 0.45
FAR_CONTACT_LEAD_SECONDS = 0.12  # median far-hitter lateness: +160 ms on Turin (23 pairs), +67..133 ms on US Open edits
MAX_CHAIN_GAP_SECONDS = 0.25
CHAIN_SLACK_PIXELS = 30.0
CHAIN_TURN_ALLOWANCE = 1.5


def _fit(frames, values):
    frames = np.asarray(frames, float)
    if len(frames) < 2:
        return 0.0, float(values[0])
    slope, intercept = np.polyfit(frames, values, 1)
    return float(slope), float(intercept)


def _line(track, tail: bool):
    frames = track["frames"][-FIT_POINTS:] if tail else track["frames"][:FIT_POINTS]
    xy = np.asarray(track["xy"][-FIT_POINTS:] if tail else track["xy"][:FIT_POINTS], float)
    return _fit(frames, xy[:, 0]), _fit(frames, xy[:, 1])


def _grown(box, fraction=HITTER_BOX_GROWTH):
    x1, y1, x2, y2 = box[:4]
    grow = fraction * (y2 - y1)
    return x1 - grow, y1 - grow, x2 + grow, y2 + grow


def _inside(point, box):
    x1, y1, x2, y2 = _grown(box)
    return bool(x1 <= point[0] <= x2 and y1 <= point[1] <= y2)


def _meeting(a, b):
    """Frame where the extrapolated tail of `a` and head of `b` are closest, with that distance."""
    (axm, axb), (aym, ayb) = _line(a, True)
    (bxm, bxb), (bym, byb) = _line(b, False)
    best = None
    for t in range(a["frames"][-1] - 3, b["frames"][0] + 4):
        pa = np.array([axm * t + axb, aym * t + ayb])
        pb = np.array([bxm * t + bxb, bym * t + byb])
        d = float(np.linalg.norm(pa - pb))
        if best is None or d < best[1]:
            best = (t, d, (pa + pb) / 2)
    return best  # None when the two tracklets overlap in time by more than the search margin


def split_at_reversals(tracks, fps, window=4):
    """Cut tracklets where the vertical image velocity reverses sharply (racket contact seen inside one track)."""
    min_speed = MIN_SPEED_PX_PER_SECOND / fps
    result = []
    for track in tracks:
        frames, xy = track["frames"], track["xy"]
        cuts, last_cut = [], -window
        j = window
        while j <= len(frames) - window - 1:
            left = _fit(frames[j - window: j + 1], [p[1] for p in xy[j - window: j + 1]])[0]
            right = _fit(frames[j: j + window + 1], [p[1] for p in xy[j: j + window + 1]])[0]
            if left * right < 0 and abs(left) >= min_speed and abs(right) >= min_speed and j - last_cut >= window:
                # Take the strongest reversal in the next few points, not the first one that qualifies.
                best_j, best_score = j, abs(left - right)
                for k in range(j + 1, min(j + window, len(frames) - window - 1) + 1):
                    l2 = _fit(frames[k - window: k + 1], [p[1] for p in xy[k - window: k + 1]])[0]
                    r2 = _fit(frames[k: k + window + 1], [p[1] for p in xy[k: k + window + 1]])[0]
                    if l2 * r2 < 0 and abs(l2 - r2) > best_score:
                        best_j, best_score = k, abs(l2 - r2)
                cuts.append(best_j)
                last_cut = best_j
                j = best_j + window
            else:
                j += 1
        edges = [0, *[c + 1 for c in cuts], len(frames)]
        for start, stop in zip(edges[:-1], edges[1:]):
            if stop - start >= 2:
                result.append({"frames": frames[start:stop], "xy": xy[start:stop]})
    return result


def _net_dy(track):
    """Net vertical image displacement and per-frame rate over the whole tracklet (up the image is negative)."""
    span = max(1, track["frames"][-1] - track["frames"][0])
    dy = track["xy"][-1][1] - track["xy"][0][1]
    return float(dy), float(dy) / span


def prepare_tracks(tracks, fps):
    """Tracklets cut at sharp reversals and without stubs; `contact["outgoing"]` indexes into this list."""
    return [t for t in split_at_reversals(tracks, fps) if len(t["frames"]) >= MIN_TRACK_POINTS // 2]


def find_contacts(tracks, fps, boxes_at, segment_start, prepared=False):
    """Contact candidates: dicts with frame, hitter, confidence and reasons."""
    min_speed = MIN_SPEED_PX_PER_SECOND / fps * 0.4
    max_gap = MAX_CONTACT_GAP_SECONDS * fps
    usable = tracks if prepared else prepare_tracks(tracks, fps)
    contacts = []
    for index, b in enumerate(usable):
        # Direction comes from the whole flight: a far hitter's ball first rises, so its head velocity is ~0.
        net_b, rate_b = _net_dy(b)
        if abs(net_b) < MIN_NET_DISPLACEMENT or abs(rate_b) < min_speed:
            continue
        hitter = "near" if net_b < 0 else "far"
        chosen = None
        for a_index in range(index - 1, -1, -1):
            a = usable[a_index]
            gap = b["frames"][0] - a["frames"][-1]
            if gap > max_gap:
                break
            net_a, rate_a = _net_dy(a)
            if abs(net_a) < MIN_NET_DISPLACEMENT or abs(rate_a) < min_speed or np.sign(net_a) == np.sign(net_b):
                continue
            meeting_result = _meeting(a, b)
            if meeting_result is None:
                continue
            frame, distance, meeting = meeting_result
            if distance > MAX_MEETING_DISTANCE:
                continue
            chosen = (a, frame, distance, meeting, a_index)
            break
        reasons, hitter_box = [], boxes_at(b["frames"][0], hitter)
        if chosen:
            a, frame, distance, meeting, _ = chosen
            box = boxes_at(frame, hitter)
            near_player = None if box is None else _inside(meeting, box)
            confidence = 0.45 * max(0.0, 1 - distance / 40) + 0.25 * min(1.0, min(len(a["frames"]), len(b["frames"])) / 14)
            confidence += {True: 0.3, None: 0.1, False: 0.0}[near_player]
            reasons.append(f"flights_meet_{distance:.0f}px_gap_{b['frames'][0] - a['frames'][-1]}f")
            reasons.append({True: "ball_at_hitter", None: "hitter_box_missing", False: "ball_not_at_hitter"}[near_player])
            if near_player is False:
                confidence *= 0.5
        else:
            # No usable incoming flight (common: the ball is lost around the bounce). A substantial flight that
            # starts at a player and travels away from them is that player's shot.
            earlier = [t for t in usable[:index] if b["frames"][0] - t["frames"][-1] < 1.5 * fps]
            if hitter_box is None or len(b["frames"]) < HITTER_FLIGHT_MIN_POINTS or abs(net_b) < HITTER_FLIGHT_MIN_NET:
                continue
            centre = np.array([(hitter_box[0] + hitter_box[2]) / 2, (hitter_box[1] + hitter_box[3]) / 2])
            height = hitter_box[3] - hitter_box[1]
            first = np.array(b["xy"][0])
            distance = float(np.linalg.norm(first - centre))
            if distance > HITTER_FLIGHT_REACH * height:
                continue
            (vx, _), (vy, _) = _line(b, False)
            travel = max(0.0, distance - 0.35 * height) / max(1.0, float(np.hypot(vx, vy)))
            frame = b["frames"][0] - int(min(round(0.5 * fps), round(travel)))
            at_player = _inside(first, hitter_box)
            confidence = (0.5 if at_player else 0.38) + (0.07 if len(b["frames"]) >= 15 else 0.0)
            reasons.append(f"flight_starts_at_hitter_{distance:.0f}px" if at_player else f"flight_starts_near_hitter_{distance:.0f}px")
            if earlier:
                reasons.append("incoming_flight_not_matched")
        contacts.append({
            "frame": int(frame), "hitter": hitter, "confidence": float(min(1.0, confidence)),
            "reasons": reasons, "outgoing": index, "incoming": chosen[4] if chosen else None,
            "first_in_segment": not chosen and not earlier,
        })
    contacts.sort(key=lambda c: c["frame"])
    kept = []
    for contact in contacts:
        if kept and contact["frame"] - kept[-1]["frame"] < DUPLICATE_CONTACT_SECONDS * fps:
            if contact["confidence"] > kept[-1]["confidence"]:
                kept[-1] = contact
            continue
        kept.append(contact)
    return kept


def find_bounce(points, fps):
    """Best slope-drop kink in image y(t): returns (frame, x, y, confidence, drop) or None.

    `points` are time-ordered (frame, x, y) detections of one flight; gaps are not filled.
    """
    if len(points) < 7:
        return None
    pts = np.asarray(points, float)
    min_drop = MIN_BOUNCE_DROP_PX_PER_SECOND / fps
    best = None
    for j in range(3, len(pts) - 3):
        left = pts[max(0, j - FIT_POINTS + 1): j + 1]
        right = pts[j: j + FIT_POINTS]
        if len(left) < 4 or len(right) < 4 or right[-1, 0] - right[0, 0] < 2 or left[-1, 0] - left[0, 0] < 2:
            continue
        ml, bl = _fit(left[:, 0], left[:, 2])
        mr, br = _fit(right[:, 0], right[:, 2])
        drop = ml - mr
        if drop < min_drop:
            continue
        residual = np.sqrt(np.mean((np.polyval([ml, bl], left[:, 0]) - left[:, 2]) ** 2)
                           + np.mean((np.polyval([mr, br], right[:, 0]) - right[:, 2]) ** 2))
        score = drop / (1.0 + residual)
        if best is None or score > best[0]:
            best = (score, j, ml, bl, mr, br, drop, residual)
    if best is None:
        return None
    _, j, ml, bl, mr, br, drop, residual = best
    # Refine with one continuous two-piece line (a kink), which is better posed than two separate fits
    # when detections thin out right at the ground.
    window = pts[max(0, j - 8): j + 9]
    kink_frame, kink_y, sse = float(pts[j, 0]), None, None
    for candidate in np.arange(pts[j, 0] - 4, pts[j, 0] + 4.01, 0.5):
        before = window[:, 0] < candidate
        design = np.column_stack([
            np.ones(len(window)), (window[:, 0] - candidate) * before, (window[:, 0] - candidate) * ~before,
        ])
        if before.sum() < 3 or (~before).sum() < 3:
            continue
        coefficients, *_ = np.linalg.lstsq(design, window[:, 2], rcond=None)
        error = float(np.sum((design @ coefficients - window[:, 2]) ** 2))
        if coefficients[1] - coefficients[2] >= min_drop and (sse is None or error < sse):
            kink_frame, kink_y, sse = float(candidate), float(coefficients[0]), error
    if kink_y is None:
        kink_frame = float(np.clip((br - bl) / (ml - mr) if abs(ml - mr) > 1e-9 else pts[j, 0], pts[j, 0] - 2, pts[j, 0] + 2))
        kink_y = ml * kink_frame + bl
    mx, bx = _fit(window[:, 0], window[:, 1])
    x, y = mx * kink_frame + bx, kink_y
    confidence = float(min(1.0, (drop * fps / 600) * (1 / (1 + residual / 4))))
    return int(round(kink_frame)), float(x), float(y), confidence, float(drop)


def _landing(point, corners, hitter):
    court = Court([list(map(float, c)) for c in corners])
    x, y = court.normalized(*point)
    try:
        return zone_for(x, y, hitter), [x, y]
    except ValueError:
        return None, [x, y]


def _overrun_frames(fps):
    return int(round(BOUNCE_OVERRUN_SECONDS * fps))


def follow_flight(pieces, start, banned, limit, fps):
    """Indices of the pieces that carry one flight on from piece `start` (a bounce splits it in two).

    Each step takes the nearest piece that begins soon after the current one ends, close to where the current
    one was heading. Unrelated noise tracklets in the same seconds are not on this path, so they never join.
    """
    chain = [start]
    max_gap = MAX_CHAIN_GAP_SECONDS * fps
    while pieces[chain[-1]]["frames"][-1] < limit:
        last = pieces[chain[-1]]
        (vx, bx), (vy, by) = _line(last, True)
        speed = float(np.hypot(vx, vy))
        best = None
        for index, piece in enumerate(pieces):
            gap = piece["frames"][0] - last["frames"][-1]
            if index in banned or index in chain or not 1 <= gap <= max_gap:
                continue
            first_frame = piece["frames"][0]
            miss = float(np.hypot(piece["xy"][0][0] - (vx * first_frame + bx), piece["xy"][0][1] - (vy * first_frame + by)))
            # After a bounce the ball turns back up the image, so the straight-line prediction drifts with the gap.
            if miss <= CHAIN_SLACK_PIXELS + CHAIN_TURN_ALLOWANCE * speed * gap and (best is None or miss < best[0]):
                best = (miss, index)
        if best is None:
            break
        chain.append(best[1])
    return chain


def propose_shots(tracks, fps, corners_at, boxes_at, segment, video_id, ball_window_seconds=0.3):
    """Proposals for one court-view segment (start, end)."""
    start, end = segment
    pieces = prepare_tracks(tracks, fps)
    contacts = find_contacts(pieces, fps, boxes_at, start, prepared=True)
    proposals = []
    for number, contact in enumerate(contacts):
        c = contact["frame"]
        horizon = c + int(MAX_BOUNCE_SEARCH_SECONDS * fps)
        following = contacts[number + 1] if number + 1 < len(contacts) else None
        # The ball often bounces just before the receiver hits it, so the search may run a little past the next
        # contact estimate; that shot's own outgoing flight is excluded instead.
        limit = min(horizon, (following["frame"] + _overrun_frames(fps)) if following else end, end)
        banned = {following["outgoing"]} if following is not None else set()
        chain = follow_flight(pieces, contact["outgoing"], banned, limit, fps)
        flight = {}
        for index in chain:
            track = pieces[index]
            for frame, (x, y) in zip(track["frames"], track["xy"]):
                if c + 2 <= frame <= limit and frame not in flight:
                    flight[frame] = (frame, x, y)
        points = [flight[f] for f in sorted(flight)]
        bounce = find_bounce(points, fps)
        bounce_frame = landing_pixel = zone = None
        bounce_conf, reasons = 0.0, list(contact["reasons"])
        if bounce is not None:
            bounce_frame, bx, by, bounce_conf, drop = bounce
            bounce_corners = corners_at(bounce_frame)
            if bounce_corners is not None:
                landing_pixel = [round(bx, 1), round(by, 1)]
                zone, normalized = _landing(landing_pixel, bounce_corners, contact["hitter"])
                reasons.append(f"bounce_slope_drop_{drop:.1f}px_per_frame")
                if zone is None:
                    reasons.append("landing_outside_court_or_wrong_side")
                    bounce_conf *= 0.4
            else:
                bounce_frame = landing_pixel = None
                bounce_conf = 0.0
                reasons.append("bounce_not_in_court_view")
        elif points:
            # No kink found: point the reviewer at where the visible flight ends, with no landing or zone guess.
            bounce_frame, bounce_conf = int(points[-1][0]) + 1, 0.0
            reasons.append("bounce_hint_from_end_of_visible_flight")
        else:
            reasons.append("no_bounce_found")
        # Measured on labeled Turin shots, kink-based bounces were early and rarely gave the right zone, so a
        # proposed bounce never reaches the confidence that allows one-key acceptance.
        bounce_conf = min(bounce_conf, BOUNCE_CONFIDENCE_CAP)
        if bounce_frame is not None:
            reasons.append("bounce_unvalidated_click_the_landing")
        trail_start, trail_end = c - int(ball_window_seconds * fps), (bounce_frame or c) + int(ball_window_seconds * fps)
        # Only this shot's own flights: the incoming ball and the outgoing flight followed through its bounce.
        shown = ([contact["incoming"]] if contact["incoming"] is not None else []) + chain
        trail = [
            [int(f), round(x, 1), round(y, 1)]
            for index in dict.fromkeys(shown) for f, (x, y) in zip(pieces[index]["frames"], pieces[index]["xy"])
            if trail_start <= f <= trail_end
        ]
        trail.sort()
        # Far-hitter contacts come out late (the ball is first tracked well after the racket); near hitters do not.
        # The id keeps the engine's own frame so decisions recorded against it stay valid.
        reported = c - (round(FAR_CONTACT_LEAD_SECONDS * fps) if contact["hitter"] == "far" else 0)
        reported = max(reported, start)
        if reported != c:
            reasons.append(f"far_contact_moved_{c - reported}f_earlier")
        corners = corners_at(reported)
        proposals.append({
            "id": f"{video_id}-{c}", "contact_frame": reported, "hitter": contact["hitter"],
            "hitter_confidence": round(contact["confidence"], 3),
            "shot_type": "serve" if contact["first_in_segment"] else "groundstroke",
            "confidence": round(contact["confidence"], 3),
            "bounce_frame": bounce_frame, "bounce_confidence": round(bounce_conf, 3),
            "landing_pixel": landing_pixel, "zone": zone,
            "contact_corners": None if corners is None else [[round(float(v), 2) for v in p] for p in corners],
            "bounce_corners": None if bounce_frame is None or corners_at(bounce_frame) is None
            else [[round(float(v), 2) for v in p] for p in corners_at(bounce_frame)],
            "players": {
                side: None if boxes_at(c, side) is None else [round(float(v), 1) for v in boxes_at(c, side)[:4]]
                for side in ("near", "far")
            },
            "ball_trail": trail, "reasons": reasons,
        })
    return proposals
