"use strict";
// Pure review-queue logic: no DOM, so it can be unit tested with node and reused by review.js.
(function (root) {
  const CONFIDENCE_GATE = 0.5;
  const MAX_GAP_STEPS = 1.5; // must match shot_predict.autopilot.camera.MAX_GAP_STEPS
  const LANDING_TOLERANCE = 0.5; // must match shot_predict.autopilot.review.LANDING_TOLERANCE_PIXELS
  const BALL_PIXELS_PER_SECOND = 3600; // must match shot_predict.autopilot.ball.GATE_PIXELS_PER_SECOND
  const UNIT_SQUARE = [[0, 1], [1, 1], [1, 0], [0, 0]]; // near-left, near-right, far-right, far-left

  function solveLinear(matrix, vector) {
    const n = vector.length;
    const a = matrix.map((row, i) => [...row, vector[i]]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
      if (Math.abs(a[pivot][col]) < 1e-12) throw new Error("Degenerate court corners.");
      [a[col], a[pivot]] = [a[pivot], a[col]];
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = a[r][col] / a[col][col];
        for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
      }
    }
    return a.map((row, i) => row[n] / row[i]);
  }

  // Homography (3x3, row-major, h[8] = 1) mapping four source points onto four destination points.
  function homography(src, dst) {
    const rows = [], values = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [u, v] = dst[i];
      rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
      values.push(u);
      rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
      values.push(v);
    }
    return [...solveLinear(rows, values), 1];
  }

  function applyHomography(h, point) {
    const [x, y] = point;
    const w = h[6] * x + h[7] * y + h[8];
    return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
  }

  const courtToImage = (corners) => homography(UNIT_SQUARE, corners);

  function courtPoint(corners, x, y) {
    return applyHomography(courtToImage(corners), [x, y]);
  }

  function gridLines(corners) {
    const h = courtToImage(corners);
    const lines = [];
    for (const x of [0, 1 / 3, 2 / 3, 1]) lines.push([applyHomography(h, [x, 0]), applyHomography(h, [x, 1])]);
    for (const y of [0, 0.25, 0.5, 0.75, 1]) lines.push([applyHomography(h, [0, y]), applyHomography(h, [1, y])]);
    return lines;
  }

  // Receiver-relative landing zones, identical to shot_predict.court.zone_for.
  function zoneRect(zone, hitter) {
    const [lateral, depth] = zone.split("_");
    let left = {ad: 0, middle: 1 / 3, deuce: 2 / 3}[lateral];
    if (left === undefined || (depth !== "deep" && depth !== "short")) throw new Error(`Unknown zone ${zone}`);
    let right = left + 1 / 3;
    if (hitter === "near") [left, right] = [1 - right, 1 - left];
    const [top, bottom] = hitter === "near"
      ? (depth === "deep" ? [0, 0.25] : [0.25, 0.5])
      : (depth === "deep" ? [0.75, 1] : [0.5, 0.75]);
    return [[left, top], [right, top], [right, bottom], [left, bottom]];
  }

  function zonePolygon(corners, zone, hitter) {
    const h = courtToImage(corners);
    return zoneRect(zone, hitter).map((p) => applyHomography(h, p));
  }

  // Court corners at a frame, interpolated exactly like shot_predict.autopilot.camera.CameraTrack.
  function interpolateCorners(camera, frame) {
    if (!camera || !camera.samples || !camera.samples.length) return null;
    const samples = camera.samples.filter((s) => s.corners).sort((a, b) => a.frame - b.frame);
    const maxGap = MAX_GAP_STEPS * camera.step_frames;
    let run = [];
    const runs = [];
    for (const s of samples) {
      const last = run[run.length - 1];
      if (last && last.reference === s.reference && s.frame - last.frame <= maxGap) run.push(s);
      else { if (run.length) runs.push(run); run = [s]; }
    }
    if (run.length) runs.push(run);
    const segment = runs.find((r) => frame >= r[0].frame && frame <= r[r.length - 1].frame);
    if (!segment) return null;
    let index = 0;
    while (index + 1 < segment.length && segment[index + 1].frame <= frame) index++;
    if (index >= segment.length - 1) return segment[segment.length - 1].corners.map((p) => [...p]);
    const a = segment[index], b = segment[index + 1];
    const weight = (frame - a.frame) / (b.frame - a.frame);
    return a.corners.map((p, i) => [(1 - weight) * p[0] + weight * b.corners[i][0], (1 - weight) * p[1] + weight * b.corners[i][1]]);
  }

  function lowConfidence(proposal, step) {
    if (step === "contact") return !(proposal.confidence >= CONFIDENCE_GATE);
    return !(proposal.bounce_confidence >= CONFIDENCE_GATE) || !proposal.landing_pixel;
  }

  function editedFields(proposal, current) {
    const edited = [];
    if (current.contact_frame !== proposal.contact_frame) edited.push("contact_frame");
    if (current.bounce_frame !== proposal.bounce_frame) edited.push("bounce_frame");
    if (current.hitter !== proposal.hitter) edited.push("hitter");
    if (current.shot_type !== proposal.shot_type) edited.push("shot_type");
    const a = current.landing_pixel, b = proposal.landing_pixel;
    if (!a || !b || Math.abs(a[0] - b[0]) > LANDING_TOLERANCE || Math.abs(a[1] - b[1]) > LANDING_TOLERANCE) edited.push("landing");
    return edited;
  }

  // Next proposal after a frame, wrapping to the earliest one; `skipped` holds ids to leave out this session.
  function nextProposal(queue, afterFrame, skipped = new Set()) {
    const open = queue.filter((p) => !skipped.has(p.id)).sort((a, b) => a.contact_frame - b.contact_frame);
    return open.find((p) => p.contact_frame > afterFrame) || open[0] || null;
  }

  function neighbour(queue, currentId, direction) {
    const open = [...queue].sort((a, b) => a.contact_frame - b.contact_frame);
    const index = open.findIndex((p) => p.id === currentId);
    if (index < 0) return open[0] || null;
    return open[Math.max(0, Math.min(open.length - 1, index + direction))];
  }

  function createSession() {
    return {asIs: 0, corrected: 0, rejected: 0, seconds: []};
  }

  function record(session, outcome, seconds) {
    if (outcome === "as_is") session.asIs++;
    else if (outcome === "corrected") session.corrected++;
    else session.rejected++;
    if ((outcome === "as_is" || outcome === "corrected") && Number.isFinite(seconds)) session.seconds.push(seconds);
  }

  // Reverse a recorded outcome (used when the human undoes the label or decision).
  function unrecord(session, outcome, seconds) {
    if (outcome === "as_is") session.asIs = Math.max(0, session.asIs - 1);
    else if (outcome === "corrected") session.corrected = Math.max(0, session.corrected - 1);
    else session.rejected = Math.max(0, session.rejected - 1);
    if (outcome === "as_is" || outcome === "corrected") {
      const index = session.seconds.lastIndexOf(seconds);
      if (index >= 0) session.seconds.splice(index, 1);
    }
  }

  const meanSeconds = (session) => (session.seconds.length ? session.seconds.reduce((a, b) => a + b, 0) / session.seconds.length : null);

  // Split a [frame, x, y] trail wherever the jump is faster than a ball can travel (reach pixels per frame).
  function trailPieces(trail, reach) {
    const pieces = [];
    let current = [];
    for (const point of trail) {
      const last = current[current.length - 1];
      if (last && Math.hypot(point[1] - last[1], point[2] - last[2]) > reach * Math.max(1, point[0] - last[0])) {
        pieces.push(current);
        current = [];
      }
      current.push(point);
    }
    if (current.length) pieces.push(current);
    return pieces;
  }

  // Whether a timer that accumulates only while visible should count: pure helper used with injected clocks.
  function createTimer(now) {
    let accumulated = 0, since = null;
    return {
      start() { accumulated = 0; since = now(); },
      pause() { if (since !== null) { accumulated += now() - since; since = null; } },
      resume() { if (since === null) since = now(); },
      seconds() { return (accumulated + (since === null ? 0 : now() - since)) / 1000; },
    };
  }

  // ----- validate mode (pure helpers; the DOM side lives in validate.js) -----
  const FLUSH_COUNT = 8;
  const FLUSH_MS = 2500;
  const AUDIT_BASE = 10;
  const AUDIT_SPREAD = 6;
  const MAX_CARD_SECONDS = 120;

  function hashString(text) {
    let hash = 2166136261;
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  const zoneLabel = (zone) => zone.replace("_", " ").toUpperCase();

  // A clearly wrong zone for an audit card: the other side of the court and the other depth.
  function decoyZone(zone, id) {
    const [lateral, depth] = zone.split("_");
    const other = lateral === "ad" ? "deuce" : lateral === "deuce" ? "ad" : (hashString(id) % 2 ? "ad" : "deuce");
    return `${other}_${depth === "deep" ? "short" : "deep"}`;
  }

  // Audits come after at least AUDIT_BASE ordinary cards plus a little id-derived jitter, so they cannot be predicted.
  const auditDue = (cardsSinceAudit, id) => cardsSinceAudit >= AUDIT_BASE + (hashString(id) % AUDIT_SPREAD);

  function cardsFor(queue, decided) {
    return queue.filter((p) => p.validatable && !decided.has(p.id)).sort((a, b) => a.contact_frame - b.contact_frame);
  }

  // The moment to loop: a little run-up before contact through just after the bounce.
  function loopWindow(proposal, fps, frames) {
    return {
      start: Math.max(0, proposal.contact_frame - Math.round(0.6 * fps)),
      end: Math.min(frames - 1, proposal.bounce_frame + Math.round(0.35 * fps)),
    };
  }

  // Next clip that still has cards, after the open one in the server's watching order (wrapping around).
  function nextClip(inbox, currentId, excluded = new Set()) {
    const open = ((inbox && inbox.clips) || []).filter((c) => c.pending > 0 && c.video_id !== currentId && !excluded.has(c.video_id));
    if (!open.length) return null;
    const order = Number.isFinite(inbox.current_order) ? inbox.current_order : -1;
    return (open.find((c) => c.order > order) || open[0]).video_id;
  }

  function flushDue(count, oldestAt, now, force = false) {
    return count > 0 && (force || count >= FLUSH_COUNT || (oldestAt !== null && now - oldestAt >= FLUSH_MS));
  }

  function makeDecision(videoId, proposalId, verdict, seconds) {
    const decision = {video_id: videoId, proposal_id: proposalId, verdict};
    if (verdict === "correct" && Number.isFinite(seconds)) decision.review_seconds = Math.min(Math.max(seconds, 0), MAX_CARD_SECONDS);
    return decision;
  }

  function createValidateStats() {
    return {confirmed: 0, wrong: 0, skipped: 0, seconds: [], auditsShown: 0, auditsMissed: 0};
  }

  function recordVerdict(stats, verdict, seconds) {
    if (verdict === "correct") stats.confirmed++;
    else if (verdict === "wrong") stats.wrong++;
    else stats.skipped++;
    if (Number.isFinite(seconds)) stats.seconds.push(Math.min(seconds, MAX_CARD_SECONDS));
  }

  function unrecordVerdict(stats, verdict, seconds) {
    if (verdict === "correct") stats.confirmed = Math.max(0, stats.confirmed - 1);
    else if (verdict === "wrong") stats.wrong = Math.max(0, stats.wrong - 1);
    else stats.skipped = Math.max(0, stats.skipped - 1);
    const index = stats.seconds.lastIndexOf(Math.min(seconds, MAX_CARD_SECONDS));
    if (index >= 0) stats.seconds.splice(index, 1);
  }

  const cardsPerMinute = (stats) => {
    const total = stats.seconds.reduce((a, b) => a + b, 0);
    return stats.seconds.length >= 3 && total > 0 ? (60 * stats.seconds.length) / total : null;
  };

  const api = {
    CONFIDENCE_GATE, homography, applyHomography, courtPoint, gridLines, zoneRect, zonePolygon, interpolateCorners,
    lowConfidence, editedFields, nextProposal, neighbour, createSession, record, unrecord, meanSeconds, createTimer,
    trailPieces, BALL_PIXELS_PER_SECOND,
    hashString, zoneLabel, decoyZone, auditDue, cardsFor, loopWindow, nextClip, flushDue, makeDecision,
    createValidateStats, recordVerdict, unrecordVerdict, cardsPerMinute, FLUSH_COUNT, FLUSH_MS,
  };
  root.ReviewLogic = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
