"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const L = require(path.join(__dirname, "..", "shot_predict", "labeler_frontend", "review_logic.js"));

const fixture = process.env.REVIEW_FIXTURE ? JSON.parse(fs.readFileSync(process.env.REVIEW_FIXTURE, "utf8")) : null;
const close = (a, b, tolerance = 0.05) => assert.ok(Math.abs(a - b) <= tolerance, `${a} vs ${b}`);

test("homography maps the court corners onto the image corners and matches Python's Court", { skip: !fixture }, () => {
  const corners = fixture.corners;
  for (const [i, [x, y]] of [[0, 1], [1, 1], [1, 0], [0, 0]].entries()) {
    const point = L.courtPoint(corners, x, y);
    close(point[0], corners[i][0], 1e-6);
    close(point[1], corners[i][1], 1e-6);
  }
  for (const item of fixture.pixels) {
    const point = L.courtPoint(corners, item.x, item.y);
    close(point[0], item.pixel[0], 0.6);
    close(point[1], item.pixel[1], 0.6);
  }
  const lines = L.gridLines(corners);
  assert.equal(lines.length, 9);
  close(lines[0][0][0], corners[3][0], 1e-6);
});

test("zone rectangles are centred on Python's zone targets for both hitters", { skip: !fixture }, () => {
  for (const item of fixture.targets) {
    const polygon = L.zonePolygon(fixture.corners, item.zone, item.hitter);
    const [cx, cy] = L.courtPoint(fixture.corners, ...centreOf(L.zoneRect(item.zone, item.hitter)));
    close(cx, item.target[0], 0.6);
    close(cy, item.target[1], 0.6);
    assert.equal(polygon.length, 4);
  }
});

function centreOf(rect) {
  return [(rect[0][0] + rect[1][0]) / 2, (rect[0][1] + rect[3][1]) / 2];
}

test("camera interpolation reproduces CameraTrack inside views and refuses cuts", { skip: !fixture }, () => {
  for (const item of fixture.camera.expected) {
    const corners = L.interpolateCorners(fixture.camera.track, item.frame);
    if (item.corners === null) {
      assert.equal(corners, null, `frame ${item.frame}`);
    } else {
      corners.forEach((p, i) => { close(p[0], item.corners[i][0], 1e-6); close(p[1], item.corners[i][1], 1e-6); });
    }
  }
});

test("confidence gating forces explicit confirmation of weak proposals", () => {
  const strong = {confidence: 0.8, bounce_confidence: 0.6, landing_pixel: [1, 2]};
  assert.equal(L.lowConfidence(strong, "contact"), false);
  assert.equal(L.lowConfidence(strong, "bounce"), false);
  assert.equal(L.lowConfidence({...strong, confidence: 0.49}, "contact"), true);
  assert.equal(L.lowConfidence({...strong, bounce_confidence: 0.45}, "bounce"), true);
  assert.equal(L.lowConfidence({...strong, landing_pixel: null}, "bounce"), true);
  assert.equal(L.lowConfidence({}, "contact"), true);
  assert.equal(L.lowConfidence({confidence: NaN}, "contact"), true);
});

test("edits are detected per field and tiny landing differences are ignored", () => {
  const proposal = {contact_frame: 100, bounce_frame: 130, hitter: "near", shot_type: "groundstroke", landing_pixel: [500, 300]};
  const same = {contact_frame: 100, bounce_frame: 130, hitter: "near", shot_type: "groundstroke", landing_pixel: [500.3, 299.8]};
  assert.deepEqual(L.editedFields(proposal, same), []);
  assert.deepEqual(L.editedFields(proposal, {...same, contact_frame: 101, hitter: "far"}), ["contact_frame", "hitter"]);
  assert.deepEqual(L.editedFields(proposal, {...same, landing_pixel: [520, 300]}), ["landing"]);
  assert.deepEqual(L.editedFields({...proposal, landing_pixel: null}, same), ["landing"]);
  assert.deepEqual(L.editedFields({...proposal, bounce_frame: null}, same), ["bounce_frame"]);
});

test("a ball trail is split wherever the jump is faster than a ball can fly", () => {
  const reach = L.BALL_PIXELS_PER_SECOND / 60; // 60 px per frame
  const flight = [[10, 100, 100], [11, 130, 110], [12, 160, 120], [14, 220, 140]];
  assert.deepEqual(L.trailPieces(flight, reach), [flight]);
  const tangled = [[10, 100, 100], [11, 130, 110], [11, 600, 500], [12, 160, 120], [13, 190, 130]];
  assert.deepEqual(L.trailPieces(tangled, reach).map((piece) => piece.length), [2, 1, 2]);
  assert.deepEqual(L.trailPieces([], reach), []);
});

test("audit decoys are clearly wrong zones and audits are spaced out", () => {
  for (const zone of ["ad_deep", "ad_short", "middle_deep", "middle_short", "deuce_deep", "deuce_short"]) {
    const decoy = L.decoyZone(zone, "clip-123");
    const [lateral, depth] = zone.split("_"), [decoyLateral, decoyDepth] = decoy.split("_");
    assert.notEqual(decoyLateral, lateral, zone);
    assert.notEqual(decoyDepth, depth, zone);
    assert.ok(["ad", "middle", "deuce"].includes(decoyLateral));
  }
  assert.equal(L.zoneLabel("deuce_short"), "DEUCE SHORT");
  for (const id of ["a-1", "b-22", "c-333", "x"]) {
    assert.equal(L.auditDue(9, id), false);
    assert.equal(L.auditDue(15, id), true);
    assert.equal(L.auditDue(15, id), L.auditDue(15, id));
  }
  assert.ok(new Set(["a-1", "b-22", "c-333", "d-4444", "e", "f"].map((id) => L.auditDue(12, id))).size === 2, "jitter should vary by id");
});

test("cards are the validatable undecided proposals in time order, and the loop brackets the shot", () => {
  const queue = [
    {id: "c", contact_frame: 300, bounce_frame: 330, validatable: true},
    {id: "a", contact_frame: 100, bounce_frame: 130, validatable: true},
    {id: "n", contact_frame: 200, bounce_frame: null, validatable: false},
    {id: "b", contact_frame: 150, bounce_frame: 180, validatable: true},
  ];
  assert.deepEqual(L.cardsFor(queue, new Set(["b"])).map((p) => p.id), ["a", "c"]);
  assert.deepEqual(L.cardsFor(queue, new Set()).map((p) => p.id), ["a", "b", "c"]);
  assert.deepEqual(L.loopWindow(queue[1], 60, 5000), {start: 64, end: 151});
  assert.deepEqual(L.loopWindow({contact_frame: 10, bounce_frame: 4990}, 60, 5000), {start: 0, end: 4999});
});

test("the next clip follows the watching order, wraps, and skips exhausted or excluded clips", () => {
  const clips = [{video_id: "a", pending: 3, order: 0}, {video_id: "c", pending: 2, order: 2}, {video_id: "d", pending: 5, order: 3}];
  const at = (current_order) => ({clips, current_order});
  assert.equal(L.nextClip(at(0), "a"), "c");
  assert.equal(L.nextClip(at(2), "c"), "d");
  assert.equal(L.nextClip(at(3), "d"), "a");
  assert.equal(L.nextClip(at(1), "b"), "c"); // the open clip is exhausted, so it is not in the list but its position is known
  assert.equal(L.nextClip(at(9), "zzz"), "a");
  assert.equal(L.nextClip({clips}, "zzz"), "a");
  assert.equal(L.nextClip(at(0), "a", new Set(["c"])), "d");
  assert.equal(L.nextClip({clips: [clips[0]], current_order: 0}, "a"), null);
  assert.equal(L.nextClip({clips: [], current_order: 0}, "a"), null);
  assert.equal(L.nextClip(null, "a"), null);
});

test("decisions are flushed on count, age or force, and carry a capped review time only when confirmed", () => {
  assert.equal(L.flushDue(0, null, 99999, true), false);
  assert.equal(L.flushDue(1, 1000, 1500), false);
  assert.equal(L.flushDue(1, 1000, 3600), true);
  assert.equal(L.flushDue(L.FLUSH_COUNT, 1000, 1001), true);
  assert.equal(L.flushDue(1, 1000, 1001, true), true);
  assert.deepEqual(L.makeDecision("v", "p", "correct", 3.2), {video_id: "v", proposal_id: "p", verdict: "correct", review_seconds: 3.2});
  assert.equal(L.makeDecision("v", "p", "correct", 9999).review_seconds, 120);
  assert.deepEqual(L.makeDecision("v", "p", "wrong", 3), {video_id: "v", proposal_id: "p", verdict: "wrong"});
});

test("validate statistics can be undone and report cards per minute only with enough data", () => {
  const stats = L.createValidateStats();
  L.recordVerdict(stats, "correct", 2);
  L.recordVerdict(stats, "wrong", 3);
  assert.equal(L.cardsPerMinute(stats), null);
  L.recordVerdict(stats, "correct", 5);
  L.recordVerdict(stats, "skip", 2);
  assert.deepEqual([stats.confirmed, stats.wrong, stats.skipped], [2, 1, 1]);
  assert.equal(Math.round(L.cardsPerMinute(stats)), 20);
  L.unrecordVerdict(stats, "correct", 5);
  L.unrecordVerdict(stats, "wrong", 3);
  assert.deepEqual([stats.confirmed, stats.wrong, stats.skipped, stats.seconds.length], [1, 0, 1, 2]);
  L.unrecordVerdict(stats, "wrong", 3);
  assert.equal(stats.wrong, 0);
});

test("queue navigation advances in time order, wraps, and honours skips", () => {
  const queue = [{id: "c", contact_frame: 300}, {id: "a", contact_frame: 100}, {id: "b", contact_frame: 200}];
  assert.equal(L.nextProposal(queue, -1).id, "a");
  assert.equal(L.nextProposal(queue, 100).id, "b");
  assert.equal(L.nextProposal(queue, 300).id, "a");
  assert.equal(L.nextProposal(queue, 100, new Set(["b"])).id, "c");
  assert.equal(L.nextProposal([], 0), null);
  assert.equal(L.nextProposal(queue, 0, new Set(["a", "b", "c"])), null);
  assert.equal(L.neighbour(queue, "b", 1).id, "c");
  assert.equal(L.neighbour(queue, "a", -1).id, "a");
  assert.equal(L.neighbour(queue, "missing", 1).id, "a");
});

test("session statistics and the visible-time timer", () => {
  const session = L.createSession();
  L.record(session, "as_is", 3);
  L.record(session, "corrected", 7);
  L.record(session, "rejected", 99);
  assert.deepEqual([session.asIs, session.corrected, session.rejected], [1, 1, 1]);
  assert.equal(L.meanSeconds(session), 5);
  assert.equal(L.meanSeconds(L.createSession()), null);
  L.unrecord(session, "corrected", 7);
  L.unrecord(session, "rejected", 0);
  assert.deepEqual([session.asIs, session.corrected, session.rejected], [1, 0, 0]);
  assert.equal(L.meanSeconds(session), 3);
  L.unrecord(session, "corrected", 7);
  L.unrecord(session, "as_is", 3);
  L.unrecord(session, "as_is", 3);
  assert.deepEqual([session.asIs, session.corrected, session.rejected, session.seconds.length], [0, 0, 0, 0]);
  L.record(session, "as_is", 3);
  L.record(session, "corrected", 7);
  L.record(session, "rejected", 99);
  let clock = 0;
  const timer = L.createTimer(() => clock);
  timer.start();
  clock = 4000;
  timer.pause();
  clock = 60000;
  assert.equal(timer.seconds(), 4);
  timer.resume();
  clock = 62000;
  assert.equal(timer.seconds(), 6);
  timer.start();
  assert.equal(timer.seconds(), 0);
});

test("degenerate corners are reported instead of producing garbage", () => {
  assert.throws(() => L.gridLines([[0, 0], [0, 0], [0, 0], [0, 0]]), /Degenerate/);
  assert.throws(() => L.zoneRect("nowhere_deep", "near"), /Unknown zone/);
});
