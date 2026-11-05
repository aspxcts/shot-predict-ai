"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "shot_predict", "labeler_frontend", "labeler.js"),
  "utf8",
);

function workspace({frameCallbacks = true} = {}) {
  const elements = new Map();
  const listeners = new Map();
  const timers = new Map();
  let nextTimer = 0;
  let nextFrame = null;
  const context2d = new Proxy({}, {get: () => () => {}});

  class Element {
    constructor() {
      this.value = "";
      this.clientWidth = 320;
      this.clientHeight = 180;
      this.style = {};
      this.classList = {toggle() {}};
      this.listeners = new Map();
    }
    addEventListener(name, callback) {
      if (!this.listeners.has(name)) this.listeners.set(name, []);
      this.listeners.get(name).push(callback);
    }
    dispatch(name, event = {}) {
      for (const callback of this.listeners.get(name) || []) callback(event);
    }
    getContext() { return context2d; }
    replaceChildren() {}
    setAttribute() {}
  }

  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const video = element("video");
  Object.assign(video, {
    readyState: 4, seeking: false, paused: true, muted: false, currentTime: 0,
    load() {},
    pause() { this.paused = true; this.dispatch("pause"); },
  });
  if (frameCallbacks) {
    video.requestVideoFrameCallback = (callback) => { nextFrame = callback; return 1; };
    video.cancelVideoFrameCallback = () => {};
  }
  element("speed").value = "1";
  const document = {
    getElementById: element,
    createElement: () => new Element(),
    addEventListener() {},
    activeElement: null,
  };
  const window = {
    parent: {innerHeight: 720, postMessage() {}, addEventListener() {}},
    screen: {availHeight: 720},
    devicePixelRatio: 1,
    addEventListener(name, callback) { listeners.set(name, callback); },
  };
  const sandbox = vm.createContext({
    document, window, console,
    ResizeObserver: class { observe() {} },
    HTMLSelectElement: Element,
    HTMLInputElement: Element,
    setTimeout(callback, delay) {
      const id = ++nextTimer;
      timers.set(id, {callback, delay});
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(source, sandbox, {filename: "labeler.js"});

  const args = {
    fps: 25, frames: 120, width: 320, height: 180, corners: [[0, 0]],
    grid: [], videos: [{id: "zero", name: "zero"}, {id: "offset", name: "offset"}],
    shot_count: 0, total_shots: 0, last_shot_id: null, acknowledgement: null,
  };
  return {
    element,
    video,
    load(id, origin, deliverFrame = true) {
      listeners.get("message")({
        data: {type: "streamlit:render", args: {...args, video_id: id, video_url: `/${id}`}},
      });
      video.currentTime = origin;
      video.dispatch("loadeddata");
      if (frameCallbacks && deliverFrame) this.display(origin);
    },
    display(mediaTime) {
      assert.ok(nextFrame, "A displayed-frame callback should be registered");
      nextFrame(0, {mediaTime});
    },
    runTimers(delay) {
      for (const [id, timer] of [...timers]) {
        if (timer.delay === delay) {
          timers.delete(id);
          timer.callback();
        }
      }
    },
    seek(frame) {
      const input = element("frame-input");
      input.value = String(frame);
      input.dispatch("change");
    },
    status() { return element("frame-status").textContent; },
  };
}

test("frame callbacks use the first presentation timestamp as frame zero", () => {
  const ui = workspace();
  ui.load("offset", 0.04);
  assert.equal(ui.status(), "Frame 0");
  for (const frame of [1, 25, 77, 119]) {
    ui.seek(frame);
    assert.ok(Math.abs(ui.video.currentTime - (0.04 + (frame + 0.5) / 25)) < 1e-9);
    ui.display(0.04 + frame / 25);
    assert.equal(ui.status(), `Frame ${frame}`);
  }
  ui.load("zero", 0);
  assert.equal(ui.status(), "Frame 0");
  ui.seek(1);
  assert.ok(Math.abs(ui.video.currentTime - 0.06) < 1e-9);
  ui.display(0.04);
  assert.equal(ui.status(), "Frame 1");
});

test("clock fallback preserves a nonzero first-frame timestamp", () => {
  const ui = workspace({frameCallbacks: false});
  ui.load("offset", 0.04);
  assert.equal(ui.status(), "Frame 0");
  assert.match(ui.element("notice").textContent, /does not support displayed-frame callbacks/);
  for (const frame of [1, 37, 119]) {
    ui.seek(frame);
    ui.video.dispatch("seeked");
    assert.equal(ui.status(), `Frame ${frame}`);
  }
});

test("delayed frame callbacks use the initial video clock as fallback", () => {
  const ui = workspace();
  ui.load("offset", 0.04, false);
  ui.runTimers(500);
  assert.equal(ui.status(), "Frame 0");
  ui.seek(37);
  ui.display(0.04 + 37 / 25);
  assert.equal(ui.status(), "Frame 37");
});

test("invalid exact-frame entries report an error without seeking", () => {
  const ui = workspace();
  ui.load("offset", 0.04);
  for (const value of ["", "-1", "1.5", "120"]) {
    ui.seek(value);
    assert.match(ui.element("notice").textContent, /Enter a whole frame from 0 to 119/);
    assert.equal(ui.element("frame-input").value, "0");
    assert.equal(ui.video.currentTime, 0.04);
    assert.equal(ui.status(), "Frame 0");
  }
});
