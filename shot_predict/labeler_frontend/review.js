"use strict";
// Review-queue mode: jump to machine-proposed shots, let the human confirm or correct them.
// Hooks into labeler.js (globals: state, video, seekFrame, stepFrames, sendAction, tell, ...) at a few points.
(function () {
  const L = window.ReviewLogic;
  const byId = (id) => document.getElementById(id);
  const review = {
    payload: null, mode: null, current: null, step: "contact", draft: null, hitterTouched: false,
    session: L.createSession(), lastFrame: -1, inFlight: null, history: [], wantFrame: null, focusId: null,
    preview: null, timer: L.createTimer(() => performance.now()),
  };
  window.__review = review; // exposed for debugging and browser tests

  const queue = () => (review.payload && review.payload.queue) || [];
  const camera = () => (review.payload && review.payload.camera) || null;
  const isActive = () => {
    if (!review.payload || !state.args) return false;
    if (window.validateActive && window.validateActive()) return false; // validate mode owns the screen
    return review.mode === null ? queue().length > 0 : review.mode;
  };
  const ready = () => state.haveFrame && state.targetFrame === null && !state.pending;
  // A proposal's jump is queued in wantFrame until the labeler can seek; acting before it lands would label the wrong frame.
  const settled = () => ready() && review.wantFrame === null;
  const percent = (value) => `${Math.round((value || 0) * 100)}%`;

  function enter(proposal) {
    cancelPreview(false);
    review.current = proposal;
    review.step = "contact";
    review.hitterTouched = false;
    review.draft = {
      contact_frame: proposal.contact_frame, bounce_frame: proposal.bounce_frame, hitter: proposal.hitter,
      shot_type: proposal.shot_type, landing_pixel: proposal.landing_pixel ? [...proposal.landing_pixel] : null,
    };
    state.hit = null;
    state.pendingPixel = null;
    review.timer.start();
    if (document.hidden) review.timer.pause();
    review.wantFrame = proposal.contact_frame;
    tell("Check the contact frame, then press Enter. Use the arrow keys to nudge.");
  }

  function ensureCurrent() {
    if (!isActive()) return;
    const items = queue();
    if (review.focusId) {
      const wanted = items.find((p) => p.id === review.focusId);
      review.focusId = null;
      if (wanted) { enter(wanted); return; }
    }
    if (review.current) {
      const same = items.find((p) => p.id === review.current.id);
      if (same) { review.current = same; return; }
    }
    const next = L.nextProposal(items, review.lastFrame);
    if (next) enter(next);
    else { review.current = null; state.hit = null; }
  }

  function trySeek() {
    if (review.wantFrame === null || !ready()) return;
    const frame = review.wantFrame;
    review.wantFrame = null;
    if (state.frame !== frame) seekFrame(frame);
  }

  function acceptContact() {
    if (!settled() || !review.current) return;
    const p = review.current;
    if (L.lowConfidence(p, "contact") && !review.hitterTouched) {
      tell("Low confidence: confirm the hitter with N or F first.", "error");
      return;
    }
    review.draft.contact_frame = state.frame;
    review.step = "bounce";
    state.hit = {frame: state.frame, hitter: review.draft.hitter, shotType: review.draft.shot_type};
    const fps = state.args.fps;
    const proposed = p.bounce_frame !== null && p.bounce_frame > state.frame ? p.bounce_frame : state.frame + Math.round(0.6 * fps);
    review.wantFrame = Math.min(state.args.frames - 1, proposed);
    tell(L.lowConfidence(p, "bounce")
      ? "The bounce is only a hint: move to the first bounce and click where the ball landed."
      : "Press Enter to accept the proposed landing, or click where the ball landed.");
    updateControls();
  }

  function submit(landing, bounceFrame) {
    const p = review.current;
    const draft = review.draft;
    if (bounceFrame <= draft.contact_frame) {
      tell("Move forward to the first bounce after contact.", "error");
      return;
    }
    draft.bounce_frame = bounceFrame;
    draft.landing_pixel = landing;
    review.inFlight = {id: p.id, edited: L.editedFields(p, draft), seconds: review.timer.seconds(), frame: p.contact_frame};
    state.pendingPixel = landing;
    drawOverlay();
    sendAction("accept_proposal", {
      proposal_id: p.id, contact_frame: draft.contact_frame, bounce_frame: bounceFrame, hitter: draft.hitter,
      shot_type: draft.shot_type, landing_pixel: landing, review_seconds: review.inFlight.seconds,
    });
  }

  function acceptBounce() {
    if (!settled() || !review.current) return;
    const p = review.current;
    if (L.lowConfidence(p, "bounce")) {
      tell("Not confident enough to accept: click where the ball landed.", "error");
      return;
    }
    submit(p.landing_pixel, state.frame);
  }

  function decide(decision) {
    if (!review.current || state.pending) return;
    cancelPreview(false);
    review.inFlight = {id: review.current.id, decision, frame: review.current.contact_frame};
    sendAction("reject_proposal", {proposal_id: review.current.id, decision});
  }

  function setHitter(side) {
    if (!review.current) return;
    review.draft.hitter = side;
    review.hitterTouched = true;
    if (state.hit) state.hit.hitter = side;
    tell(`${side === "near" ? "Near" : "Far"} hitter.`);
    updateControls();
    drawOverlay();
  }

  function setType(type) {
    if (!review.current) return;
    review.draft.shot_type = type;
    if (state.hit) state.hit.shotType = type;
    updateControls();
  }

  function back() {
    if (!review.current || review.step !== "bounce") return;
    review.step = "contact";
    state.hit = null;
    review.wantFrame = review.draft.contact_frame;
    updateControls();
  }

  function move(direction) {
    const target = L.neighbour(queue(), review.current && review.current.id, direction);
    if (target && (!review.current || target.id !== review.current.id)) {
      enter(target);
      updateControls();
    }
  }

  function toManual() {
    cancelPreview(false);
    review.mode = false;
    state.hit = null;
    tell("Manual labeling. Switch the review queue back on in the header.");
    updateControls();
    drawOverlay();
  }

  function toggleMode() {
    if (!review.payload) return;
    if (isActive()) { toManual(); return; }
    review.mode = true;
    if (!queue().length) tell(review.payload.message || "The review queue is empty.");
    ensureCurrent();
    updateControls();
    drawOverlay();
  }

  function cancelPreview(finish) {
    const p = review.preview;
    if (!p) return;
    clearInterval(p.timer);
    video.pause();
    video.playbackRate = Number(byId("speed").value);
    review.preview = null;
    if (finish) review.wantFrame = p.to;
  }

  function preview() {
    if (!review.current || !ready()) return;
    cancelPreview(false);
    const fps = state.args.fps;
    const from = review.step === "contact" ? state.frame : review.draft.contact_frame;
    const proposedBounce = review.current.bounce_frame;
    const to = Math.min(state.args.frames - 1, (proposedBounce !== null && proposedBounce > from ? proposedBounce : from + Math.round(0.8 * fps)));
    seekFrame(from);
    const entry = {to, started: false, timer: null};
    entry.timer = setInterval(() => {
      if (state.targetFrame !== null || !state.haveFrame) return;
      if (!entry.started) {
        entry.started = true;
        video.playbackRate = 0.25;
        video.play().catch(() => cancelPreview(false));
        return;
      }
      if (state.frame >= entry.to || video.paused) cancelPreview(true);
    }, 40);
    review.preview = entry;
    tell("Slow-motion preview.");
  }

  function undoReview() {
    const last = review.history[review.history.length - 1];
    if (!last) { tell("Nothing to undo in this session."); return; }
    if (last.kind === "review") sendAction("undo_review", {proposal_id: last.proposalId});
    else if (state.args.last_shot_id) sendAction("undo", {shot_id: state.args.last_shot_id});
  }

  // ----- hooks called from labeler.js -----
  window.reviewReset = function () {
    cancelPreview(false);
    review.current = null;
    review.mode = null;
    review.inFlight = null;
    review.history = [];
    review.lastFrame = -1;
    review.focusId = null;
    if (window.validateReset) window.validateReset();
  };

  window.reviewRender = function (payload) {
    review.payload = payload || null;
    ensureCurrent();
    if (window.validateRender) window.validateRender();
  };

  // Open one proposal in the detailed contact -> bounce flow (used by "Fix it" in validate mode).
  window.reviewFocus = function (id) {
    review.mode = true;
    review.focusId = id;
    ensureCurrent();
    updateControls();
    drawOverlay();
  };

  window.reviewAck = function (ack) {
    if (!ack) return;
    if (window.validateAck) window.validateAck(ack);
    const flight = review.inFlight;
    if (ack.action === "accept_proposal") {
      state.pendingPixel = null;
      if (ack.ok && flight) {
        const edited = (ack.edited_fields || []).length > 0;
        const outcome = edited ? "corrected" : "as_is";
        L.record(review.session, outcome, flight.seconds);
        review.history.push({kind: "shot", proposalId: flight.id, outcome, seconds: flight.seconds});
        review.lastFrame = flight.frame;
        review.current = null;
        state.hit = null;
      }
      review.inFlight = null;
    } else if (ack.action === "reject_proposal") {
      if (ack.ok && flight) {
        L.record(review.session, "rejected", 0);
        review.history.push({kind: "review", proposalId: flight.id, outcome: "rejected", seconds: 0});
        review.lastFrame = flight.frame;
        review.current = null;
        state.hit = null;
      }
      review.inFlight = null;
    } else if (ack.action === "undo_review" && ack.ok) {
      const last = review.history.pop();
      if (last) {
        L.unrecord(review.session, last.outcome, last.seconds);
        review.focusId = last.proposalId;
      }
    } else if (ack.action === "undo" && ack.ok) {
      const index = review.history.map((h) => h.kind).lastIndexOf("shot");
      if (index >= 0) {
        const entry = review.history.splice(index, 1)[0];
        L.unrecord(review.session, entry.outcome, entry.seconds);
        review.focusId = entry.proposalId;
      }
    }
  };

  window.reviewClick = function (point) {
    if (window.validateActive && window.validateActive()) return true; // clicks do nothing while validating
    if (!isActive() || state.calibrating || !review.current) return false;
    if (!settled()) {
      tell("Still jumping to the proposed frame. Try again in a moment.");
      return true;
    }
    if (review.step === "contact") {
      tell("First accept the contact with Enter (or set the hitter with N / F).");
      return true;
    }
    submit(point, state.frame);
    return true;
  };

  window.reviewOverlay = function (ctx, pos) {
    if (window.validateOverlay && window.validateOverlay(ctx, pos)) return;
    if (!isActive() || !review.current || state.calibrating) return;
    const p = review.current;
    const corners = L.interpolateCorners(camera(), state.frame) || (camera() ? null : (state.args.corners || p.contact_corners));
    const line = (points, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      points.forEach((pt, i) => { const [x, y] = pos(pt); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke();
    };
    if (camera() && corners && state.gridVisible) {
      for (const segment of L.gridLines(corners)) line(segment, "rgba(70, 235, 125, .88)", 2);
    }
    if (Math.abs(state.frame - p.contact_frame) <= 4) {
      for (const side of ["near", "far"]) {
        const box = p.players && p.players[side];
        if (!box) continue;
        const [x1, y1] = pos([box[0], box[1]]);
        const [x2, y2] = pos([box[2], box[3]]);
        const hitter = side === review.draft.hitter;
        ctx.strokeStyle = side === "near" ? "#46c8f5" : "#5ee68b";
        ctx.lineWidth = hitter ? 4 : 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = "bold 13px sans-serif";
        ctx.fillText(`${side.toUpperCase()}${hitter ? " - HITTER" : ""}`, x1, Math.max(14, y1 - 6));
      }
    }
    const trail = (p.ball_trail || []).filter((t) => t[0] <= state.frame && state.frame - t[0] <= 45);
    if (trail.length) {
      // Join only points a ball could have travelled between (the tracker's own speed limit); never zig-zag across the court.
      const reach = L.BALL_PIXELS_PER_SECOND / state.args.fps;
      const pieces = L.trailPieces(trail, reach);
      for (const piece of pieces) {
        if (piece.length < 2) continue; // a lone point is not a flight
        line(piece.map((t) => [t[1], t[2]]), "rgba(255, 228, 92, .95)", 3);
        for (const t of piece) {
          const [x, y] = pos([t[1], t[2]]);
          ctx.fillStyle = t[0] === state.frame ? "#fff" : "#ffe45c";
          ctx.beginPath();
          ctx.arc(x, y, t[0] === state.frame ? 5 : 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
    // A landing the engine itself flagged as outside the court (no zone) is not worth showing.
    if (review.step === "bounce" && p.landing_pixel && p.zone) {
      if (corners) {
        const polygon = L.zonePolygon(corners, p.zone, review.draft.hitter);
        ctx.fillStyle = "rgba(230, 70, 235, .22)";
        ctx.beginPath();
        polygon.forEach((pt, i) => { const [x, y] = pos(pt); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
        ctx.closePath();
        ctx.fill();
      }
      const [x, y] = pos(p.landing_pixel);
      ctx.strokeStyle = L.lowConfidence(p, "bounce") ? "#ffb347" : "#e646eb";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, 2 * Math.PI);
      ctx.moveTo(x - 17, y); ctx.lineTo(x + 17, y); ctx.moveTo(x, y - 17); ctx.lineTo(x, y + 17);
      ctx.stroke();
    }
    const low = L.lowConfidence(p, review.step);
    const label = review.step === "contact"
      ? `CONTACT? ${review.draft.hitter.toUpperCase()} | ${review.draft.shot_type} | ${percent(p.confidence)}`
      : `BOUNCE? ${p.zone ? p.zone.replace("_", " ") + " | " : ""}${percent(p.bounce_confidence)} (hint)`;
    ctx.font = "bold 16px sans-serif";
    const width = ctx.measureText(label).width + 20;
    ctx.fillStyle = "rgba(16, 26, 33, .85)";
    ctx.fillRect(10, 10, width, 30);
    ctx.fillStyle = low ? "#ffb347" : "#7dffc4";
    ctx.fillText(label, 20, 31);
  };

  window.reviewSync = function () {
    const active = isActive() && !state.calibrating;
    byId("review-panel").hidden = !active;
    byId("review-shortcuts").hidden = !active;
    byId("manual-shortcuts").hidden = active;
    if (active) byId("label-panel").hidden = true;
    const toggle = byId("review-toggle");
    toggle.hidden = !review.payload;
    toggle.disabled = state.pending !== null;
    toggle.textContent = isActive() ? "Review queue: ON" : "Review queue: OFF";
    toggle.classList.toggle("on", isActive());
    const total = review.payload ? review.payload.queue_total : 0;
    const mean = L.meanSeconds(review.session);
    const stats = review.session;
    byId("review-stats").textContent = review.payload
      ? `${total} to review | as-is ${stats.asIs} | corrected ${stats.corrected} | rejected ${stats.rejected}` +
        (mean === null ? "" : ` | ${mean.toFixed(1)} s/shot`)
      : "";
    trySeek();
    // The overlay depends on the displayed frame and step, which the original code does not redraw on.
    drawOverlay();
    if (window.validateSync) window.validateSync(); // after the panels above: validate mode hides the others
    if (!active) return;
    const busy = state.pending !== null;
    const p = review.current;
    const panelButtons = ["review-accept", "review-near", "review-far", "review-groundstroke", "review-serve",
      "review-preview", "review-back", "review-reject", "review-out", "review-skip"];
    if (!p) {
      byId("review-eyebrow").textContent = "REVIEW QUEUE";
      byId("review-title").textContent = "Queue empty";
      byId("review-chips").replaceChildren();
      byId("review-help").textContent = (review.payload.message || "Every proposal for this clip has been reviewed.") +
        (review.payload.dropped ? ` (${review.payload.dropped} malformed proposals ignored.)` : "");
      for (const id of panelButtons) byId(id).disabled = true;
      byId("review-manual").disabled = busy;
      return;
    }
    const index = queue().findIndex((item) => item.id === p.id);
    byId("review-eyebrow").textContent = `REVIEW ${index + 1} / ${queue().length}${total > queue().length ? ` (${total} total)` : ""}`;
    const contact = review.step === "contact";
    const low = L.lowConfidence(p, review.step);
    byId("review-title").textContent = contact ? "Contact?" : "Bounce?";
    byId("review-help").textContent = contact
      ? (low ? "Weak proposal. Check the frame, confirm the hitter with N or F, then Enter."
             : "Is the ball leaving the racket? Enter to accept, arrows to nudge.")
      : (low ? "The bounce is a hint. Find the first bounce, then click where the ball landed."
             : "Enter accepts the proposed landing; clicking sets a different one.");
    const chips = [
      [`${review.draft.hitter === "near" ? "Near" : "Far"} hitter`, ""],
      [review.draft.shot_type, ""],
      [`contact ${percent(p.confidence)}`, p.confidence >= L.CONFIDENCE_GATE ? "" : "amber"],
    ];
    if (!contact) chips.push([`bounce ${percent(p.bounce_confidence)}`, p.bounce_confidence >= L.CONFIDENCE_GATE ? "" : "amber"]);
    byId("review-chips").replaceChildren(...chips.map(([text, kind]) => {
      const chip = document.createElement("span");
      chip.className = `chip ${kind}`;
      chip.textContent = text;
      return chip;
    }));
    const acceptLabel = contact ? "Accept contact" : (low ? "Click the landing" : "Accept landing");
    byId("review-accept").firstChild.textContent = acceptLabel + " ";
    byId("review-accept").disabled = busy || !state.haveFrame || (low && !contact) || (low && contact && !review.hitterTouched);
    for (const id of ["review-reject", "review-out", "review-skip", "review-near", "review-far", "review-groundstroke", "review-serve", "review-preview"]) {
      byId(id).disabled = busy;
    }
    byId("review-back").disabled = busy || contact;
    byId("review-manual").disabled = busy;
    byId("review-near").classList.toggle("selected", review.draft.hitter === "near");
    byId("review-far").classList.toggle("selected", review.draft.hitter === "far");
    byId("review-groundstroke").classList.toggle("selected", review.draft.shot_type === "groundstroke");
    byId("review-serve").classList.toggle("selected", review.draft.shot_type === "serve");
  };

  // ----- events -----
  document.addEventListener("keydown", (event) => {
    if (!isActive() || state.calibrating || !state.args || state.pending) return;
    const target = event.target;
    if (target instanceof HTMLSelectElement || (target instanceof HTMLInputElement && target !== timeline)) return;
    const key = event.key;
    const lower = key.toLowerCase();
    const handled = (action) => { event.preventDefault(); event.stopImmediatePropagation(); action(); };
    if ((event.ctrlKey || event.metaKey) && lower === "z") return handled(undoReview);
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (lower === "m") return handled(toManual);
    if (lower === "v" && window.validateEnter) return handled(window.validateEnter);
    if (!review.current) return;
    if (key === "Enter") return handled(() => (review.step === "contact" ? acceptContact() : acceptBounce()));
    if (lower === "n") return handled(() => setHitter("near"));
    if (lower === "f") return handled(() => setHitter("far"));
    if (lower === "g") return handled(() => setType("groundstroke"));
    if (lower === "s") return handled(() => setType("serve"));
    if (lower === "x") return handled(() => decide("rejected"));
    if (lower === "k") return handled(() => decide("out"));
    if (lower === "q") return handled(() => decide("skipped"));
    if (lower === "p") return handled(preview);
    if (key === "Backspace") return handled(back);
    if (key === "[") return handled(() => move(-1));
    if (key === "]") return handled(() => move(1));
    // Shift + "," / "." arrives as "<" / ">" on most keyboards.
    const nudge = {",": -1, "<": -1, ".": 1, ">": 1}[key];
    if (review.step === "bounce" && nudge !== undefined) {
      return handled(() => stepFrames(nudge * (event.shiftKey ? 5 : 1)));
    }
    if (review.preview && (key === " " || key.startsWith("Arrow"))) cancelPreview(false);
  }, true);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) review.timer.pause(); else if (review.current) review.timer.resume();
  });

  const wire = (id, fn) => byId(id).addEventListener("click", fn);
  wire("review-toggle", toggleMode);
  wire("review-accept", () => (review.step === "contact" ? acceptContact() : acceptBounce()));
  wire("review-near", () => setHitter("near"));
  wire("review-far", () => setHitter("far"));
  wire("review-groundstroke", () => setType("groundstroke"));
  wire("review-serve", () => setType("serve"));
  wire("review-preview", preview);
  wire("review-back", back);
  wire("review-reject", () => decide("rejected"));
  wire("review-out", () => decide("out"));
  wire("review-skip", () => decide("skipped"));
  wire("review-manual", toManual);
})();
