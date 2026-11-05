"use strict";

const byId = (id) => document.getElementById(id);
const video = byId("video");
const canvas = byId("overlay");
const context = canvas.getContext("2d");
const timeline = byId("timeline");
const frameInput = byId("frame-input");
const clipSelect = byId("clip");
const state = {
  args: null,
  videoId: null,
  videoUrl: null,
  frame: 0,
  frameOrigin: null,
  targetFrame: null,
  haveFrame: false,
  hit: null,
  shotType: "groundstroke",
  calibrating: false,
  points: [],
  calibrationFrame: null,
  pendingPixel: null,
  hover: null,
  pending: null,
  lastAck: null,
  gridVisible: true,
  generation: 0,
  callbackId: null,
  seekTimer: null,
  videoList: "",
  notice: "",
  noticeKind: "",
};
const supportsFrameCallback = typeof video.requestVideoFrameCallback === "function";
let eventSequence = 0;
let lastHeight = 0;

function message(type, data) {
  window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");
}

function fitHeight() {
  let height;
  try {
    height = window.parent.innerHeight;
  } catch (error) {
    console.warn("Cannot read the parent viewport; using the screen height.", error);
    height = window.screen.availHeight;
  }
  height = Math.max(480, Math.round(height - 65));
  if (height !== lastHeight) {
    lastHeight = height;
    message("streamlit:setFrameHeight", {height});
  }
}

function tell(text, kind = "") {
  state.notice = text;
  state.noticeKind = kind;
  byId("notice").textContent = text;
  byId("notice").className = `notice ${kind}`;
}

function clampFrame(frame) {
  return Math.max(0, Math.min(state.args.frames - 1, frame));
}

function frameFromClock(seconds) {
  return clampFrame(Math.floor((seconds - state.frameOrigin) * state.args.fps + 0.0001));
}

function frameFromPresentation(seconds) {
  return clampFrame(Math.round((seconds - state.frameOrigin) * state.args.fps));
}

function timeText(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds - minutes * 60).toFixed(2).padStart(5, "0");
  return `${String(minutes).padStart(2, "0")}:${remainder}`;
}

function isLocked() {
  return state.calibrating && state.points.length > 0;
}

function readyFrame() {
  return state.haveFrame && state.targetFrame === null;
}

function updateFrameControls() {
  if (!state.args) return;
  const frame = state.targetFrame === null ? state.frame : state.targetFrame;
  timeline.value = String(frame);
  if (document.activeElement !== frameInput) frameInput.value = String(frame);
  byId("frame-status").textContent = `Frame ${frame}${state.targetFrame === null ? "" : " (seeking)"}`;
  byId("time").textContent = timeText(frame / state.args.fps);
  byId("duration").textContent = timeText((state.args.frames - 1) / state.args.fps);
}

function acceptClockFrame(target) {
  if (!state.args || video.seeking || video.readyState < 2) return;
  if (state.frameOrigin === null) state.frameOrigin = video.currentTime;
  const frame = frameFromClock(video.currentTime);
  if (target !== null && frame !== target) {
    tell(`The browser could not reach frame ${target}. Try seeking again.`, "error");
    return;
  }
  state.frame = frame;
  state.haveFrame = true;
  state.targetFrame = null;
  clearTimeout(state.seekTimer);
  byId("screen-message").hidden = true;
  updateControls();
  drawLoupe();
}

function updateControls() {
  if (!state.args) return;
  const busy = state.pending !== null;
  const navigationDisabled = busy || !state.haveFrame || isLocked();
  for (const id of ["back-five", "back-frame", "play", "forward-frame", "forward-five", "timeline", "frame-input"]) {
    byId(id).disabled = navigationDisabled;
  }
  byId("play").textContent = video.paused ? "Play" : "Pause";
  byId("sound").textContent = video.muted ? "Sound off" : "Sound on";
  byId("speed").disabled = busy;
  byId("sound").disabled = busy;
  clipSelect.disabled = busy;
  byId("calibration-panel").hidden = !state.calibrating;
  byId("label-panel").hidden = state.calibrating;
  byId("save-court").disabled = busy || state.points.length !== 4;
  byId("reset-corners").disabled = busy || state.points.length === 0;
  byId("cancel-court").hidden = !state.args.corners;
  byId("cancel-court").disabled = busy;
  for (let i = 0; i < 4; i++) {
    byId(`corner-${i}`).className = i < state.points.length ? "done" : i === state.points.length ? "current" : "";
  }

  const marked = state.hit !== null;
  byId("choose-hit").hidden = marked;
  byId("phase").textContent = marked ? "STEP 2 OF 2" : "STEP 1 OF 2";
  byId("phase-title").textContent = marked ? "Find the first bounce" : "Find racket contact";
  byId("phase-help").textContent = marked
    ? "Find the first bounce, then click its green landing zone to save. Hover to magnify."
    : "Play or scrub to contact, then choose the hitter.";
  byId("hit-summary").textContent = marked
    ? `${state.hit.hitter === "near" ? "Near" : "Far"} hit at frame ${state.hit.frame} (${state.hit.shotType}).`
    : "No contact marked yet.";
  byId("redo-hit").hidden = !marked;
  byId("redo-hit").disabled = busy;
  for (const id of ["near", "far"]) byId(id).disabled = busy || marked || !readyFrame();
  for (const id of ["groundstroke", "serve"]) {
    const selected = id === state.shotType;
    byId(id).classList.toggle("selected", selected);
    byId(id).setAttribute("aria-pressed", String(selected));
    byId(id).disabled = busy || marked;
  }
  byId("recalibrate").disabled = busy || !state.haveFrame;
  byId("undo").disabled = busy || !state.args.last_shot_id;
  byId("grid").textContent = state.gridVisible ? "Hide court grid" : "Show court grid";
  byId("grid").disabled = busy;
  byId("label-count").textContent = `${state.args.shot_count} in clip | ${state.args.total_shots} total`;
  if (!marked || state.calibrating || busy) {
    state.hover = null;
    byId("loupe").hidden = true;
  }
  updateFrameControls();
  if (window.reviewSync) window.reviewSync();
}

function videoRectangle() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const scale = Math.min(width / state.args.width, height / state.args.height);
  return {
    scale,
    left: (width - state.args.width * scale) / 2,
    top: (height - state.args.height * scale) / 2,
  };
}

function drawOverlay() {
  if (!state.args) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const rect = videoRectangle();
  const pos = (point) => [rect.left + point[0] * rect.scale, rect.top + point[1] * rect.scale];
  if (!state.calibrating && state.gridVisible) {
    context.strokeStyle = "rgba(70, 235, 125, .88)";
    context.lineWidth = 2;
    for (const line of state.args.grid) {
      const start = pos(line[0]);
      const end = pos(line[1]);
      context.beginPath();
      context.moveTo(...start);
      context.lineTo(...end);
      context.stroke();
    }
  }
  if (state.calibrating && state.points.length) {
    context.strokeStyle = "#ffda64";
    context.lineWidth = 3;
    context.beginPath();
    state.points.forEach((point, index) => {
      const [x, y] = pos(point);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    if (state.points.length === 4) context.closePath();
    context.stroke();
    state.points.forEach((point, index) => {
      const [x, y] = pos(point);
      context.beginPath();
      context.fillStyle = "#ffda64";
      context.arc(x, y, 11, 0, 2 * Math.PI);
      context.fill();
      context.fillStyle = "#172024";
      context.font = "bold 14px sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(index + 1), x, y);
    });
  }
  if (state.pendingPixel) {
    const [x, y] = pos(state.pendingPixel);
    context.strokeStyle = "#ffda64";
    context.lineWidth = 3;
    context.beginPath();
    context.arc(x, y, 9, 0, 2 * Math.PI);
    context.stroke();
  }
  if (window.reviewOverlay) window.reviewOverlay(context, pos, rect);
}

function videoPoint(event) {
  const box = canvas.getBoundingClientRect();
  const rect = videoRectangle();
  const x = (event.clientX - box.left - rect.left) / rect.scale;
  const y = (event.clientY - box.top - rect.top) / rect.scale;
  if (x < 0 || y < 0 || x >= state.args.width || y >= state.args.height) return null;
  return [x, y];
}

function drawLoupe() {
  const loupe = byId("loupe");
  if (!state.hover || !state.hit || state.calibrating || state.pending || video.readyState < 2) {
    loupe.hidden = true;
    return;
  }
  const [x, y] = state.hover.point;
  const sourceWidth = Math.min(100, state.args.width);
  const sourceHeight = Math.min(75, state.args.height);
  const sx = Math.max(0, Math.min(state.args.width - sourceWidth, x - sourceWidth / 2));
  const sy = Math.max(0, Math.min(state.args.height - sourceHeight, y - sourceHeight / 2));
  const lens = byId("loupe-image");
  const ctx = lens.getContext("2d");
  ctx.drawImage(video, sx, sy, sourceWidth, sourceHeight, 0, 0, lens.width, lens.height);
  const px = (x - sx) * lens.width / sourceWidth;
  const py = (y - sy) * lens.height / sourceHeight;
  ctx.strokeStyle = "#ffda64";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(px - 10, py);
  ctx.lineTo(px + 10, py);
  ctx.moveTo(px, py - 10);
  ctx.lineTo(px, py + 10);
  ctx.stroke();
  loupe.style.left = state.hover.pointerX > canvas.clientWidth / 2 ? "10px" : "auto";
  loupe.style.right = state.hover.pointerX > canvas.clientWidth / 2 ? "auto" : "10px";
  loupe.style.top = state.hover.pointerY > canvas.clientHeight / 2 ? "10px" : "auto";
  loupe.style.bottom = state.hover.pointerY > canvas.clientHeight / 2 ? "auto" : "10px";
  loupe.hidden = false;
}

function watchFrames() {
  if (!supportsFrameCallback) return;
  if (state.callbackId !== null) video.cancelVideoFrameCallback(state.callbackId);
  const generation = state.generation;
  function onFrame(_now, metadata) {
    if (generation !== state.generation) return;
    if (state.frameOrigin === null) state.frameOrigin = metadata.mediaTime;
    const frame = frameFromPresentation(metadata.mediaTime);
    if (state.targetFrame === null || state.targetFrame === frame) {
      state.frame = frame;
      state.haveFrame = true;
      state.targetFrame = null;
      clearTimeout(state.seekTimer);
      byId("screen-message").hidden = true;
      updateControls();
      drawLoupe();
    }
    state.callbackId = video.requestVideoFrameCallback(onFrame);
  }
  state.callbackId = video.requestVideoFrameCallback(onFrame);
}

function seekFrame(frame) {
  if (!state.args || state.pending || !state.haveFrame) return;
  if (isLocked()) {
    tell("Reset the corner clicks before moving to another frame.", "error");
    return;
  }
  frame = clampFrame(Math.round(frame));
  video.pause();
  if (frame === state.frame && state.targetFrame === null) {
    updateFrameControls();
    return;
  }
  state.targetFrame = frame;
  video.currentTime = state.frameOrigin + (frame + 0.5) / state.args.fps;
  updateControls();
  clearTimeout(state.seekTimer);
  const generation = state.generation;
  state.seekTimer = setTimeout(() => {
    if (generation === state.generation && state.targetFrame === frame) {
      tell(`Frame ${frame} is still loading. Try seeking again or use a constant-frame-rate MP4.`, "error");
    }
  }, 2500);
}

function stepFrames(amount) {
  const from = state.targetFrame === null ? state.frame : state.targetFrame;
  seekFrame(from + amount);
}

function seekFromInput() {
  if (!state.args) return;
  const raw = frameInput.value.trim();
  const frame = Number(raw);
  if (raw === "" || !Number.isInteger(frame) || frame < 0 || frame >= state.args.frames) {
    tell(`Enter a whole frame from 0 to ${state.args.frames - 1}.`, "error");
    frameInput.value = String(state.targetFrame === null ? state.frame : state.targetFrame);
    return;
  }
  tell("Moving to the selected frame.");
  seekFrame(frame);
}

function sendAction(action, fields = {}) {
  if (state.pending) return;
  const eventId = `${Date.now()}-${++eventSequence}-${Math.random().toString(36).slice(2)}`;
  const timer = setTimeout(() => {
    if (state.pending && state.pending.id === eventId) {
      tell("Still waiting for the app. Check its connection; do not retry until it responds.", "error");
    }
  }, 10000);
  state.pending = {id: eventId, action, timer};
  tell(action === "select_video" ? "Opening clip..." : "Saving...");
  updateControls();
  message("streamlit:setComponentValue", {
    value: {event_id: eventId, action, video_id: state.videoId, ...fields},
  });
}

function markHit(hitter) {
  if (state.calibrating || state.pending || state.hit) return;
  if (!readyFrame()) {
    tell("Wait until the selected frame has finished loading.", "error");
    return;
  }
  video.pause();
  state.hit = {frame: state.frame, hitter, shotType: state.shotType};
  tell("Contact marked. Move to the first bounce and click its green landing zone.");
  updateControls();
}

function chooseType(type) {
  if (state.pending || state.hit || state.calibrating) return;
  state.shotType = type;
  updateControls();
}

function beginCalibration() {
  if (state.pending || !state.haveFrame) return;
  video.pause();
  state.calibrating = true;
  state.points = [];
  state.calibrationFrame = null;
  state.pendingPixel = null;
  state.hover = null;
  tell("Pause on a clear court view, then click the four singles corners.");
  drawOverlay();
  updateControls();
}

function undoLast() {
  if (state.pending || !state.args.last_shot_id) return;
  sendAction("undo", {shot_id: state.args.last_shot_id});
}

async function togglePlay() {
  if (state.pending || !readyFrame() || isLocked()) {
    if (isLocked()) tell("Reset the corner clicks before moving to another frame.", "error");
    return;
  }
  if (video.paused) {
    try {
      await video.play();
    } catch (error) {
      tell(`Cannot play this video: ${error.message}`, "error");
    }
  } else {
    video.pause();
  }
}

function applyAcknowledgement(ack) {
  if (!ack || ack.event_id === state.lastAck) return;
  state.lastAck = ack.event_id;
  if (!state.pending || state.pending.id !== ack.event_id) return;
  clearTimeout(state.pending.timer);
  state.pending = null;
  if (ack.ok) {
    if (ack.action === "save_shot") {
      state.hit = null;
      state.shotType = "groundstroke";
      state.pendingPixel = null;
    } else if (ack.action === "save_calibration") {
      state.calibrating = false;
      state.points = [];
      state.calibrationFrame = null;
      state.gridVisible = true;
    }
    tell(ack.message, "success");
  } else {
    state.pendingPixel = null;
    tell(ack.message, "error");
  }
  if (window.reviewAck) window.reviewAck(ack);
}

function setVideo(args) {
  state.generation += 1;
  clearTimeout(state.seekTimer);
  state.videoId = args.video_id;
  state.videoUrl = args.video_url;
  state.frame = 0;
  state.frameOrigin = null;
  state.targetFrame = null;
  state.haveFrame = false;
  state.hit = null;
  state.shotType = "groundstroke";
  state.calibrating = !args.corners && !(args.review && args.review.camera);
  if (window.reviewReset) window.reviewReset();
  state.points = [];
  state.calibrationFrame = null;
  state.pendingPixel = null;
  state.gridVisible = true;
  timeline.max = String(args.frames - 1);
  frameInput.max = String(args.frames - 1);
  video.src = args.video_url;
  video.playbackRate = Number(byId("speed").value);
  video.load();
  watchFrames();
  byId("screen-message").textContent = "Loading video...";
  byId("screen-message").hidden = false;
  tell(state.calibrating
    ? "Pause on a clear court view, then click the four singles corners."
    : "Find contact, then choose the near or far hitter.");
}

function render(args) {
  const videoList = JSON.stringify(args.videos);
  if (state.videoList !== videoList) {
    state.videoList = videoList;
    clipSelect.replaceChildren(...args.videos.map((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.name;
      return option;
    }));
  }
  state.args = args;
  if (state.videoId !== args.video_id || state.videoUrl !== args.video_url) setVideo(args);
  clipSelect.value = args.video_id;
  applyAcknowledgement(args.acknowledgement);
  if (window.reviewRender) window.reviewRender(args.review || null);
  updateControls();
  drawOverlay();
  fitHeight();
}

window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "streamlit:render") render(event.data.args);
});
window.addEventListener("resize", fitHeight);
try {
  window.parent.addEventListener("resize", fitHeight);
} catch (error) {
  console.warn("Cannot listen for parent viewport changes.", error);
}
new ResizeObserver(drawOverlay).observe(byId("screen"));

video.addEventListener("loadeddata", () => {
  if (!state.args) return;
  if (!supportsFrameCallback) {
    acceptClockFrame(null);
    tell("This browser does not support displayed-frame callbacks; use a current Chrome or Edge for frame accuracy.", "error");
  } else {
    const generation = state.generation;
    setTimeout(() => {
      if (state.generation === generation && !state.haveFrame && state.targetFrame === null) {
        acceptClockFrame(null);
      }
    }, 500);
  }
});
video.addEventListener("seeked", () => {
  if (!state.args) return;
  if (!supportsFrameCallback) {
    acceptClockFrame(state.targetFrame);
  } else if (state.targetFrame === state.frame && state.haveFrame) {
    state.targetFrame = null;
    clearTimeout(state.seekTimer);
    updateControls();
  } else if (state.targetFrame !== null) {
    const generation = state.generation;
    const target = state.targetFrame;
    setTimeout(() => {
      if (state.generation === generation && state.targetFrame === target) acceptClockFrame(target);
    }, 500);
  }
});
video.addEventListener("error", () => {
  state.haveFrame = false;
  byId("screen-message").hidden = false;
  byId("screen-message").textContent = "Video could not be played. Try an H.264 MP4 in Chrome or Edge.";
  tell("The browser cannot decode this video. Convert it to a constant-frame-rate H.264 MP4.", "error");
  updateControls();
});
video.addEventListener("play", updateControls);
video.addEventListener("pause", updateControls);
video.addEventListener("timeupdate", () => {
  if (!supportsFrameCallback && state.args && state.targetFrame === null) {
    state.frame = frameFromClock(video.currentTime);
    updateFrameControls();
  }
});

clipSelect.addEventListener("change", () => {
  const selected = clipSelect.value;
  if (selected === state.videoId) return;
  if ((state.hit || state.points.length) && !window.confirm("Switch clips and discard the unfinished label or calibration?")) {
    clipSelect.value = state.videoId;
    return;
  }
  video.pause();
  sendAction("select_video", {selected_video: selected});
});
timeline.addEventListener("input", () => seekFrame(Number(timeline.value)));
frameInput.addEventListener("change", seekFromInput);
frameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    seekFromInput();
    frameInput.blur();
  }
});
byId("back-five").addEventListener("click", () => stepFrames(-Math.round(5 * state.args.fps)));
byId("back-frame").addEventListener("click", () => stepFrames(-1));
byId("play").addEventListener("click", togglePlay);
byId("forward-frame").addEventListener("click", () => stepFrames(1));
byId("forward-five").addEventListener("click", () => stepFrames(Math.round(5 * state.args.fps)));
byId("speed").addEventListener("change", () => { video.playbackRate = Number(byId("speed").value); });
byId("sound").addEventListener("click", () => { video.muted = !video.muted; updateControls(); });
byId("groundstroke").addEventListener("click", () => chooseType("groundstroke"));
byId("serve").addEventListener("click", () => chooseType("serve"));
byId("near").addEventListener("click", () => markHit("near"));
byId("far").addEventListener("click", () => markHit("far"));
byId("redo-hit").addEventListener("click", () => {
  state.hit = null;
  tell("Hit cleared. Find contact and choose the hitter again.");
  updateControls();
});
byId("recalibrate").addEventListener("click", beginCalibration);
byId("undo").addEventListener("click", undoLast);
byId("grid").addEventListener("click", () => {
  state.gridVisible = !state.gridVisible;
  drawOverlay();
  updateControls();
});
byId("reset-corners").addEventListener("click", () => {
  state.points = [];
  state.calibrationFrame = null;
  tell("Corner clicks cleared. You can move to another frame.");
  drawOverlay();
  updateControls();
});
byId("cancel-court").addEventListener("click", () => {
  if (!state.args.corners) return;
  state.calibrating = false;
  state.points = [];
  state.calibrationFrame = null;
  tell("Kept the previous court calibration.");
  drawOverlay();
  updateControls();
});
byId("save-court").addEventListener("click", () => {
  if (state.points.length === 4) sendAction("save_calibration", {points: state.points});
});

canvas.addEventListener("click", (event) => {
  if (state.pending) return;
  if (!readyFrame()) {
    tell("Wait until the selected frame has finished loading.", "error");
    return;
  }
  const point = videoPoint(event);
  if (!point) {
    tell("Click inside the video image.", "error");
    return;
  }
  video.pause();
  if (window.reviewClick && window.reviewClick(point)) return;
  if (state.calibrating) {
    if (state.points.length >= 4) {
      tell("Save the court or reset the corner clicks.");
      return;
    }
    if (state.points.length === 0) state.calibrationFrame = state.frame;
    if (state.frame !== state.calibrationFrame) {
      tell("All four corners must be clicked on the same frame.", "error");
      return;
    }
    state.points.push(point);
    tell(state.points.length === 4 ? "Check the outline, then select Save court." : "Click the next highlighted corner.");
    drawOverlay();
    updateControls();
    return;
  }
  if (!state.hit) {
    tell("First find contact and choose the near or far hitter.");
    return;
  }
  if (state.frame <= state.hit.frame) {
    tell("Move forward to the first bounce after contact.", "error");
    return;
  }
  state.pendingPixel = point;
  drawOverlay();
  sendAction("save_shot", {
    contact_frame: state.hit.frame, bounce_frame: state.frame,
    hitter: state.hit.hitter, shot_type: state.hit.shotType, landing_pixel: point,
  });
});

canvas.addEventListener("mousemove", (event) => {
  if (!state.hit || state.calibrating || !readyFrame()) return;
  const point = videoPoint(event);
  if (!point) {
    state.hover = null;
    byId("loupe").hidden = true;
    return;
  }
  const box = canvas.getBoundingClientRect();
  state.hover = {point, pointerX: event.clientX - box.left, pointerY: event.clientY - box.top};
  drawLoupe();
});
canvas.addEventListener("mouseleave", () => {
  state.hover = null;
  byId("loupe").hidden = true;
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!state.args || state.pending || !state.haveFrame || event.deltaY === 0) return;
  const direction = event.deltaY > 0 ? 1 : -1;
  stepFrames(direction * (event.shiftKey ? Math.round(state.args.fps) : 1));
}, {passive: false});

document.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target instanceof HTMLSelectElement || (target instanceof HTMLInputElement && target !== timeline)) return;
  if (!state.args || state.pending) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoLast();
    return;
  }
  if (event.key === " ") {
    event.preventDefault();
    togglePlay();
  } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    stepFrames(direction * (event.shiftKey ? Math.round(state.args.fps) : 1));
  } else if (event.key.toLowerCase() === "j") {
    stepFrames(-Math.round(5 * state.args.fps));
  } else if (event.key.toLowerCase() === "l") {
    stepFrames(Math.round(5 * state.args.fps));
  } else if (event.key.toLowerCase() === "n") {
    markHit("near");
  } else if (event.key.toLowerCase() === "f") {
    markHit("far");
  } else if (event.key.toLowerCase() === "g") {
    chooseType("groundstroke");
  } else if (event.key.toLowerCase() === "s") {
    chooseType("serve");
  }
});

message("streamlit:componentReady", {apiVersion: 1});
fitHeight();
