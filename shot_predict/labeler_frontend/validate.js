"use strict";
// Validate mode: confirm machine-proposed shots at a glance. Each card loops the moment (run-up, contact, bounce) with
// the proposed landing zone drawn on it; the reviewer answers Correct / Wrong. Decisions are sent in batches so the
// loop never waits for the server, and cards continue across clips. Labels are built from the proposal on disk by the
// server, never from anything this file sends beyond the verdict.
(function () {
  const L = window.ReviewLogic;
  const byId = (id) => document.getElementById(id);
  const AUDIT_KEY = "shotlab.validate.audits";
  const ACTIVE_KEY = "shotlab.validate.active";
  const HOLD_MS = 450;
  const store = {
    get(key, fallback) { try { const value = window.localStorage.getItem(key); return value === null ? fallback : value; } catch (error) { return fallback; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch (error) { /* storage can be blocked in frames */ } },
  };
  const v = {
    active: false, current: null, decoy: null, focusId: null, focusUntil: 0,
    decided: new Set(), outbox: [], oldestAt: null, sending: null, retries: 0, forceFlush: false,
    history: [], undoRequest: null, stats: L.createValidateStats(), lastFrame: -1,
    cardsShown: 0, sinceAudit: 0, auditDone: new Set(), auditsOn: store.get(AUDIT_KEY, "1") === "1",
    noCards: new Set(), switching: false, allDone: false, autoEntered: false,
    loop: {phase: "seek", start: 0, end: 0, holdUntil: 0, frozen: false, playedAt: 0},
    timer: L.createTimer(() => performance.now()),
  };
  window.__validate = v; // exposed for debugging and browser tests

  const review = () => window.__review;
  const payload = () => review().payload;
  const queue = () => (payload() && payload().queue) || [];
  const camera = () => (payload() && payload().camera) || null;
  const cards = () => L.cardsFor(queue(), v.decided);
  const usable = () => v.active && !!state.args && !state.calibrating;
  const percent = (value) => `${Math.round((value || 0) * 100)}%`;

  window.validateActive = () => usable();

  // ----- cards -----
  function enterCard(proposal) {
    v.current = proposal;
    v.decoy = null;
    v.timer.start();
    if (document.hidden) v.timer.pause();
    v.cardsShown++;
    v.sinceAudit++;
    if (v.auditsOn && !v.auditDone.has(proposal.id) && v.cardsShown > 3 && L.auditDue(v.sinceAudit, proposal.id)) {
      v.decoy = {zone: L.decoyZone(proposal.zone, proposal.id)};
      v.sinceAudit = 0;
    }
    startLoop(proposal);
  }

  function startLoop(proposal) {
    const window_ = L.loopWindow(proposal, state.args.fps, state.args.frames);
    Object.assign(v.loop, {phase: "seek", start: window_.start, end: window_.end, holdUntil: 0, frozen: false, playedAt: 0});
    video.muted = true; // muted playback is always allowed to start, and the loops would otherwise repeat the audio
    video.playbackRate = Number(byId("speed").value);
  }

  function ensureCard() {
    if (!usable() || !payload()) return;
    const items = cards();
    if (v.focusId) {
      const wanted = items.find((p) => p.id === v.focusId);
      if (wanted) { v.focusId = null; v.allDone = false; enterCard(wanted); return; }
      // An undone decision returns with the next payload from the server; wait for it rather than skipping ahead.
      if (performance.now() < v.focusUntil) return;
      v.focusId = null;
    }
    if (v.current && items.some((p) => p.id === v.current.id)) {
      v.current = items.find((p) => p.id === v.current.id);
      return;
    }
    const next = L.nextProposal(items, v.lastFrame);
    if (next) {
      v.allDone = false;
      enterCard(next);
    } else {
      v.current = null;
      v.noCards.add(state.videoId);
      advanceClip();
    }
  }

  function advanceClip() {
    if (v.switching || !usable()) return;
    if (v.outbox.length || v.sending || v.undoRequest) { v.forceFlush = true; pump(); return; }
    if (state.pending) return;
    const target = L.nextClip(payload().inbox, state.videoId, v.noCards);
    if (!target) { v.allDone = true; return; }
    v.switching = true;
    v.target = target;
    v.lastFrame = -1;
    sendAction("select_video", {selected_video: target});
    if (!state.pending) v.switching = false;
  }

  // ----- the looping playback -----
  function tick() {
    if (!usable() || !v.current || v.loop.frozen || document.hidden || !state.haveFrame) return;
    const lp = v.loop;
    const now = performance.now();
    if (lp.phase === "seek") {
      if (state.targetFrame !== null || state.pending) return;
      if (Math.abs(state.frame - lp.start) <= 1) { lp.phase = "play"; lp.playedAt = now; video.play().catch(() => { lp.frozen = true; }); return; }
      seekFrame(lp.start);
    } else if (lp.phase === "play") {
      if (state.frame >= lp.end || (video.paused && now - lp.playedAt > 400)) {
        video.pause();
        lp.phase = "hold";
        lp.holdUntil = now + HOLD_MS;
      }
    } else if (lp.phase === "hold" && now >= lp.holdUntil) {
      lp.phase = "seek";
    }
  }
  setInterval(tick, 40);

  function replay() {
    if (!v.current) return;
    startLoop(v.current);
  }

  function freeze() {
    v.loop.frozen = true;
    video.pause();
  }

  function togglePause() {
    if (!v.current) return;
    if (v.loop.frozen) { v.loop.frozen = false; v.loop.phase = state.frame >= v.loop.end ? "seek" : "play"; v.loop.playedAt = performance.now(); video.play().catch(() => {}); }
    else freeze();
  }

  // ----- decisions -----
  function decide(verdict) {
    const p = v.current;
    if (!p || !usable()) return;
    if (v.decoy) { resolveAudit(verdict); return; }
    const seconds = v.timer.seconds();
    const decision = L.makeDecision(state.videoId, p.id, verdict, seconds);
    v.outbox.push(decision);
    if (v.oldestAt === null) v.oldestAt = performance.now();
    v.decided.add(p.id);
    v.history.push({decision, status: "queued", frame: p.contact_frame, seconds});
    L.recordVerdict(v.stats, verdict, seconds);
    v.lastFrame = p.contact_frame;
    v.current = null;
    ensureCard();
    pump();
    sync();
  }

  function resolveAudit(verdict) {
    const p = v.current;
    const zone = v.decoy.zone;
    v.decoy = null;
    v.auditDone.add(p.id);
    if (verdict === "correct" || verdict === "wrong") {
      const caught = verdict === "wrong";
      v.stats.auditsShown++;
      if (!caught) v.stats.auditsMissed++;
      v.outbox.push({kind: "audit", video_id: state.videoId, proposal_id: p.id, caught});
      if (v.oldestAt === null) v.oldestAt = performance.now();
      tell(caught
        ? `Audit card caught: it showed ${L.zoneLabel(zone)} on purpose. Here is the real one.`
        : `That was an audit card: ${L.zoneLabel(zone)} was deliberately wrong and you marked it correct. Check where the ball lands. Here is the real one.`,
      caught ? "success" : "error");
    }
    v.timer.start();
    startLoop(p);
    pump();
    sync();
  }

  function fixInReview() {
    const p = v.current;
    if (!p) return;
    v.auditDone.add(p.id);
    leave(false, false);
    window.reviewFocus(p.id);
  }

  function undo() {
    const last = v.history[v.history.length - 1];
    if (!last) { tell("Nothing to undo in this session."); return; }
    if (v.undoRequest || v.sending) { tell("Wait for the last save to finish, then undo."); return; }
    const verdict = last.decision.verdict;
    if (last.status === "queued") {
      const index = v.outbox.indexOf(last.decision);
      if (index >= 0) v.outbox.splice(index, 1);
      finishUndo(last);
    } else if (last.status === "failed") {
      finishUndo(last);
    } else if (last.decision.video_id !== state.videoId) {
      tell("Undo only works in the open clip. Open that clip to undo this decision.", "error");
    } else {
      v.undoRequest = last;
      v.forceFlush = true;
      if (verdict === "correct") sendUndo(last);
      else sendUndoReview(last);
    }
    sync();
  }

  function sendUndo(entry) {
    if (state.pending) return;
    if (!entry.shotId) { v.undoRequest = null; tell("That label has no saved ID to undo.", "error"); return; }
    sendAction("undo", {shot_id: entry.shotId});
  }

  function sendUndoReview(entry) {
    if (state.pending) return;
    sendAction("undo_review", {proposal_id: entry.decision.proposal_id});
  }

  function finishUndo(entry) {
    v.history.splice(v.history.indexOf(entry), 1);
    L.unrecordVerdict(v.stats, entry.decision.verdict, entry.seconds);
    v.decided.delete(entry.decision.proposal_id);
    v.lastFrame = entry.frame - 1;
    v.focusId = entry.decision.proposal_id;
    v.focusUntil = performance.now() + 4000;
    v.allDone = false;
    v.noCards.delete(state.videoId);
    v.current = null;
    ensureCard();
  }

  // ----- sending -----
  function pump() {
    if (!payload() || v.sending || state.pending) return;
    if (v.undoRequest) return;
    if (!L.flushDue(v.outbox.length, v.oldestAt, performance.now(), v.forceFlush)) return;
    const batch = v.outbox.splice(0, 60);
    v.sending = batch;
    v.oldestAt = v.outbox.length ? performance.now() : null;
    sendAction("validate_batch", {decisions: batch});
    if (!state.pending) { v.outbox.unshift(...batch); v.sending = null; }
  }
  setInterval(pump, 500);

  window.validateAck = function (ack) {
    if (!ack) return;
    if (ack.action === "save_camera_reference") {
      if (ack.ok) { // the camera is registered; leave the court-setup panel (analysis comes from the batch)
        state.calibrating = false;
        state.points = [];
        state.calibrationFrame = null;
        drawOverlay();
        updateControls();
      }
      return;
    }
    if (ack.action === "validate_batch") {
      const batch = v.sending;
      v.sending = null;
      if (!batch) return;
      if (ack.ok) {
        v.retries = 0;
        const results = ack.results || [];
        let failed = 0, firstMessage = "";
        batch.forEach((decision, index) => {
          const result = results[index];
          const entry = v.history.find((h) => h.decision === decision);
          if (result && result.ok) {
            if (entry) { entry.status = "sent"; entry.shotId = result.shot_id || null; }
          } else {
            failed++;
            firstMessage = firstMessage || (result && result.message) || "";
            if (entry) entry.status = "failed";
          }
        });
        if (failed) tell(`${failed} of ${batch.length} decisions could not be saved: ${firstMessage}`, "error");
      } else if (++v.retries <= 3) {
        v.outbox.unshift(...batch);
        v.oldestAt = v.oldestAt === null ? performance.now() : v.oldestAt;
      } else {
        v.retries = 0;
        for (const decision of batch) { const entry = v.history.find((h) => h.decision === decision); if (entry) entry.status = "failed"; }
        tell(`Could not save ${batch.length} decisions: ${ack.message}`, "error");
      }
      if (!v.outbox.length) v.forceFlush = false;
    } else if ((ack.action === "undo" || ack.action === "undo_review") && v.undoRequest) {
      const entry = v.undoRequest;
      v.undoRequest = null;
      if (ack.ok) finishUndo(entry);
    } else if (ack.action === "select_video") {
      v.switching = false;
      if (!ack.ok && v.target) v.noCards.add(v.target); // never retry a clip that could not be opened
    }
    sync();
  };

  // ----- mode changes -----
  function enter() {
    if (!payload() || !state.args) return;
    if (state.calibrating) {
      // This clip has nothing to validate (no calibration or analysis), but others may: go to the first clip with cards.
      const target = L.nextClip(payload().inbox, state.videoId, new Set());
      if (!target) { tell("Nothing to validate yet. Run the batch (see the README), then reopen this page.", "error"); return; }
      v.active = true;
      v.allDone = false;
      v.noCards.clear();
      store.set(ACTIVE_KEY, "1");
      v.target = target;
      sendAction("select_video", {selected_video: target});
      return;
    }
    v.active = true;
    v.allDone = false;
    v.noCards.clear();
    store.set(ACTIVE_KEY, "1");
    if (Number(byId("speed").value) === 1) { byId("speed").value = "0.5"; }
    video.muted = true;
    state.hit = null;
    state.pendingPixel = null;
    review().current = null;
    tell("Validate: Enter if the ball lands in the purple zone, X if it does not.");
    ensureCard();
    updateControls();
    drawOverlay();
  }

  function leave(remember = true, rerender = true) {
    if (!v.active) return;
    v.active = false;
    if (remember) store.set(ACTIVE_KEY, "0");
    video.pause();
    v.current = null;
    v.forceFlush = true;
    pump();
    if (rerender) window.reviewRender(payload());
    updateControls();
    drawOverlay();
  }

  window.validateEnter = enter;
  window.validateReset = function () {
    v.current = null;
    v.decoy = null;
    v.switching = false;
    v.lastFrame = -1;
    v.loop.phase = "seek";
    v.loop.frozen = false;
  };
  window.validateRender = function () {
    if (!payload()) return;
    if (!v.active && !v.autoEntered && store.get(ACTIVE_KEY, "0") === "1" && (cards().length || (payload().inbox && payload().inbox.pending_total > 0))) {
      v.autoEntered = true;
      enter();
      return;
    }
    if (v.active) ensureCard();
  };

  // ----- drawing -----
  window.validateOverlay = function (ctx, pos) {
    if (!usable()) return false;
    const p = v.current;
    if (!p) return true;
    const frame = state.frame, fps = state.args.fps;
    const line = (points, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      points.forEach((pt, i) => { const [x, y] = pos(pt); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke();
    };
    const corners = L.interpolateCorners(camera(), frame) || (camera() ? null : (state.args.corners || p.contact_corners));
    if (camera() && corners && state.gridVisible) {
      for (const segment of L.gridLines(corners)) line(segment, "rgba(70, 235, 125, .55)", 2);
    }
    const hitterBox = p.players && p.players[p.hitter];
    // The box was measured at contact, so show it only around that moment (the player moves away from it).
    if (hitterBox && Math.abs(frame - p.contact_frame) <= Math.max(3, Math.round(0.1 * fps))) {
      const [x1, y1] = pos([hitterBox[0], hitterBox[1]]);
      const [x2, y2] = pos([hitterBox[2], hitterBox[3]]);
      const flash = Math.abs(frame - p.contact_frame) <= Math.max(2, Math.round(0.06 * fps));
      ctx.strokeStyle = p.hitter === "near" ? "#46c8f5" : "#5ee68b";
      ctx.lineWidth = flash ? 5 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (flash) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = "bold 15px sans-serif";
        ctx.fillText("CONTACT", x1, Math.max(16, y1 - 8));
      }
    }
    const grown = (p.ball_trail || []).filter((t) => t[0] <= frame && frame - t[0] <= Math.round(1.2 * fps));
    for (const piece of L.trailPieces(grown, L.BALL_PIXELS_PER_SECOND / fps)) {
      if (piece.length < 2) continue;
      line(piece.map((t) => [t[1], t[2]]), "rgba(255, 228, 92, .95)", 3);
      const [x, y] = pos([piece[piece.length - 1][1], piece[piece.length - 1][2]]);
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    const shownZone = v.decoy ? v.decoy.zone : p.zone;
    const courtCorners = p.bounce_corners || corners;
    if (courtCorners && frame >= p.bounce_frame - 3) {
      const polygon = L.zonePolygon(courtCorners, shownZone, p.hitter);
      ctx.fillStyle = "rgba(190, 80, 245, .30)";
      ctx.strokeStyle = "rgba(225, 150, 255, .95)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      polygon.forEach((pt, i) => { const [x, y] = pos(pt); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      const centre = v.decoy ? [polygon.reduce((s, q) => s + q[0], 0) / 4, polygon.reduce((s, q) => s + q[1], 0) / 4] : p.landing_pixel;
      const [x, y] = pos(centre);
      ctx.strokeStyle = "#f7a6ff";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, 2 * Math.PI);
      ctx.moveTo(x - 17, y); ctx.lineTo(x + 17, y); ctx.moveTo(x, y - 17); ctx.lineTo(x, y + 17);
      ctx.stroke();
    }
    const label = `${p.hitter.toUpperCase()} HITS → ${L.zoneLabel(shownZone)}`;
    ctx.font = "bold 18px sans-serif";
    const width = ctx.measureText(label).width + 22;
    ctx.fillStyle = "rgba(14, 26, 33, .82)";
    ctx.fillRect(10, 10, width, 34);
    ctx.fillStyle = "#f3d9ff";
    ctx.fillText(label, 21, 33);
    return true;
  };

  // ----- panel -----
  let lastPanel = "";
  function setText(id, text) { const node = byId(id); if (node.textContent !== text) node.textContent = text; }

  window.validateSync = function () {
    byId("save-reference").disabled = state.pending !== null || state.points.length !== 4;
    const toggle = byId("validate-toggle");
    toggle.hidden = !payload();
    toggle.disabled = state.pending !== null && !v.sending;
    toggle.textContent = usable() ? "Validate: ON" : "Validate: OFF";
    toggle.classList.toggle("on", usable());
    const active = usable();
    byId("validate-panel").hidden = !active;
    byId("validate-shortcuts").hidden = !active;
    if (!active) return;
    byId("review-panel").hidden = true;
    byId("review-shortcuts").hidden = true;
    byId("label-panel").hidden = true;
    byId("manual-shortcuts").hidden = true;
    const info = payload().inbox || {};
    const inClip = cards().length + (v.current && !cards().some((p) => p.id === v.current.id) ? 1 : 0);
    const unsent = v.outbox.filter((d) => d.verdict).length + (v.sending ? v.sending.filter((d) => d.verdict).length : 0);
    const overall = Math.max(0, (info.pending_total || 0) - unsent);
    const p = v.current;
    const state_ = JSON.stringify([p && p.id, v.decoy && v.decoy.zone, inClip, overall, v.stats.confirmed, v.stats.wrong, v.stats.skipped,
      v.stats.auditsShown, v.stats.auditsMissed, v.auditsOn, v.allDone, v.switching, state.pending !== null, v.history.length]);
    if (state_ === lastPanel) return;
    lastPanel = state_;
    setText("validate-eyebrow", `VALIDATE | ${inClip} left in this clip | ${overall.toLocaleString()} overall`);
    const busy = state.pending !== null && !v.sending;
    if (p) {
      setText("validate-title", `${p.hitter === "near" ? "Near" : "Far"} hits → ${L.zoneLabel(v.decoy ? v.decoy.zone : p.zone)}`);
      setText("validate-help", "Watch the loop: does the ball land in the purple zone, and did this player just hit it?");
    } else {
      setText("validate-title", v.allDone ? "All caught up" : v.focusId ? "Bringing the card back..." : v.switching ? "Opening next clip..." : "No cards here");
      setText("validate-help", v.allDone
        ? `Nothing left to validate in ${info.analyzed_clips || 0} analyzed clips. Run more analysis, or switch to the review queue.`
        : "Moving to the next clip with cards.");
    }
    const chips = [];
    if (p) {
      chips.push([p.shot_type, ""], [`contact ${percent(p.confidence)}`, p.confidence >= L.CONFIDENCE_GATE ? "" : "amber"]);
    }
    byId("validate-chips").replaceChildren(...chips.map(([text, kind]) => {
      const chip = document.createElement("span");
      chip.className = `chip ${kind}`;
      chip.textContent = text;
      return chip;
    }));
    for (const id of ["validate-correct", "validate-wrong", "validate-edit", "validate-skip", "validate-replay"]) byId(id).disabled = !p || busy;
    byId("validate-undo").disabled = !v.history.length || !!v.undoRequest;
    const mean = L.cardsPerMinute(v.stats);
    const parts = [`<b>${v.stats.confirmed}</b> confirmed | <b>${v.stats.wrong}</b> wrong | ${v.stats.skipped} unsure`];
    if (mean !== null) parts.push(`${mean.toFixed(0)} cards/min`);
    if (v.stats.auditsShown) {
      const missed = v.stats.auditsMissed;
      parts.push(`audits ${v.stats.auditsShown - missed}/${v.stats.auditsShown} caught` + (missed ? ` <span class="bad">(${missed} missed: slow down)</span>` : ""));
    }
    const unsavedNote = unsent ? ` | ${unsent} saving` : "";
    byId("validate-stats").innerHTML = parts.join(" | ") + unsavedNote;
    setText("validate-audit", `Audit cards: ${v.auditsOn ? "ON" : "OFF"} A`);
  };

  // ----- events -----
  document.addEventListener("keydown", (event) => {
    if (!usable()) return;
    const target = event.target;
    if (target instanceof HTMLSelectElement || (target instanceof HTMLInputElement && target !== timeline)) return;
    if (event.altKey || event.metaKey || /^F\d+$/.test(event.key) || event.key === "Tab") return;
    const key = event.key;
    const lower = key.toLowerCase();
    const handled = (action) => { event.preventDefault(); event.stopImmediatePropagation(); action(); };
    if ((event.ctrlKey || event.metaKey) && lower === "z") return handled(undo);
    if (event.ctrlKey) return;
    if (state.pending && !v.sending && lower !== "z") return handled(() => {});
    if (key === "Enter" || lower === "y") return handled(() => decide("correct"));
    if (lower === "x" || lower === "n" || key === "Backspace") return handled(() => decide("wrong"));
    if (lower === "q") return handled(() => { if (v.decoy) { v.decoy = null; v.auditDone.add(v.current.id); startLoop(v.current); sync(); } else decide("skip"); });
    if (lower === "e") return handled(fixInReview);
    if (lower === "r") return handled(replay);
    if (lower === "z") return handled(undo);
    if (lower === "a") return handled(toggleAudits);
    if (lower === "v") return handled(() => leave(true));
    if (lower === "m") return handled(() => { leave(true); byId("review-manual").click(); });
    if (key === " ") return handled(togglePause);
    if (key === "ArrowLeft" || key === "ArrowRight") return handled(() => { freeze(); stepFrames(key === "ArrowLeft" ? -1 : 1); });
    handled(() => {});
  }, true);

  function toggleAudits() {
    v.auditsOn = !v.auditsOn;
    store.set(AUDIT_KEY, v.auditsOn ? "1" : "0");
    tell(v.auditsOn ? "Audit cards on: a deliberately wrong card appears now and then." : "Audit cards off.");
    sync();
  }

  function sync() { lastPanel = ""; updateControls(); }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { v.timer.pause(); v.forceFlush = true; pump(); }
    else if (v.current) v.timer.resume();
  });
  window.addEventListener("pagehide", () => { v.forceFlush = true; pump(); });

  const wire = (id, fn) => byId(id).addEventListener("click", fn);
  // Registering a camera happens on the court-setup panel, which validate mode never shows.
  wire("save-reference", () => {
    if (state.points.length === 4 && state.calibrationFrame !== null) {
      sendAction("save_camera_reference", {points: state.points, frame: state.calibrationFrame});
    }
  });
  wire("validate-toggle", () => (usable() ? leave(true) : enter()));
  wire("validate-correct", () => decide("correct"));
  wire("validate-wrong", () => decide("wrong"));
  wire("validate-skip", () => { if (v.decoy) { v.decoy = null; v.auditDone.add(v.current.id); startLoop(v.current); sync(); } else decide("skip"); });
  wire("validate-edit", fixInReview);
  wire("validate-replay", replay);
  wire("validate-undo", undo);
  wire("validate-audit", toggleAudits);
  wire("validate-exit", () => leave(true));
})();
