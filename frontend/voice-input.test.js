/**
 * Unit tests for voice-input.js's state machine, using a fake
 * SpeechRecognition and minimal fake DOM elements — no real browser or
 * microphone involved, no test framework dependency (Node's built-in
 * node:test/node:assert).
 *
 * Run with: node --test frontend/
 */
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { createVoiceInput } = require("./voice-input.js");

function makeButton() {
  const handlers = {};
  return {
    hidden: false,
    disabled: false,
    textContent: "",
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
    },
    addEventListener(type, fn) { handlers[type] = fn; },
    click() {
      if (this.disabled) return; // real <button disabled> never fires click
      handlers.click && handlers.click();
    },
  };
}

function makeEl(initial = "") {
  return { value: initial, textContent: "", hidden: false };
}

class FakeSpeechRecognition {
  constructor() {
    this._handlers = {};
    this.continuous = false;
    this.interimResults = false;
    this.lang = "";
    this.startCallCount = 0;
    this.stopCallCount = 0;
    this.abortCallCount = 0;
    this.throwOnStart = false;
  }
  addEventListener(type, fn) {
    this._handlers[type] = fn;
  }
  start() {
    this.startCallCount++;
    if (this.throwOnStart) {
      const err = new Error("recognition has already started");
      err.name = "InvalidStateError";
      throw err;
    }
  }
  stop() {
    this.stopCallCount++;
  }
  abort() {
    this.abortCallCount++;
  }
  fireStart() { this._handlers.start && this._handlers.start(); }
  fireResult(results, resultIndex = 0) {
    this._handlers.result && this._handlers.result({ results, resultIndex });
  }
  fireError(errorCode) {
    this._handlers.error && this._handlers.error({ error: errorCode });
  }
  fireEnd() { this._handlers.end && this._handlers.end(); }
}

function result(transcript, isFinal) {
  const r = [{ transcript }];
  r.isFinal = isFinal;
  return r;
}

function setup(overrides = {}) {
  const buttonEl = makeButton();
  const textareaEl = makeEl();
  const statusEl = makeEl();
  const recognitions = [];
  const SpeechRecognitionCtor = overrides.SpeechRecognitionCtor || function () {
    const r = new FakeSpeechRecognition();
    if (overrides.throwOnStart) r.throwOnStart = true;
    recognitions.push(r);
    return r;
  };
  const handle = createVoiceInput({ SpeechRecognitionCtor, buttonEl, textareaEl, statusEl });
  return { buttonEl, textareaEl, statusEl, handle, recognition: () => recognitions[0] };
}

test("unsupported browser: hides the button and explains why instead of leaving a dead control", () => {
  const buttonEl = makeButton();
  const textareaEl = makeEl();
  const statusEl = makeEl();
  const handle = createVoiceInput({ SpeechRecognitionCtor: null, buttonEl, textareaEl, statusEl });

  assert.equal(buttonEl.hidden, true);
  assert.match(statusEl.textContent, /isn't supported/);
  assert.equal(handle.getState(), "unsupported");
});

test("clicking Record starts recognition and moves to the listening state on the start event", () => {
  const { buttonEl, statusEl, handle, recognition } = setup();

  buttonEl.click();
  assert.equal(recognition().startCallCount, 1);
  assert.equal(handle.getState(), "starting");
  assert.equal(buttonEl.disabled, true); // can't double-click while starting

  recognition().fireStart();
  assert.equal(handle.getState(), "listening");
  assert.equal(buttonEl.disabled, false);
  assert.match(buttonEl.textContent, /Stop/);
  assert.equal(buttonEl.classList.contains("recording"), true);
  assert.equal(statusEl.textContent, "Listening...");
});

test("BUG FIX: clicking Stop calls abort(), not stop() — stop() is known to hang on some browsers", () => {
  const { buttonEl, handle, recognition } = setup();

  buttonEl.click();
  recognition().fireStart();
  assert.equal(handle.getState(), "listening");

  buttonEl.click(); // this is the "Stop" click
  assert.equal(recognition().abortCallCount, 1, "must use abort() for a guaranteed-immediate stop");
  assert.equal(recognition().stopCallCount, 0);
  assert.equal(handle.getState(), "stopping");
});

test("BUG FIX: the button always returns to idle/clickable after stopping, never stays stuck", () => {
  const { buttonEl, statusEl, handle, recognition } = setup();

  buttonEl.click();
  recognition().fireStart();
  buttonEl.click(); // stop
  recognition().fireError("aborted"); // real browsers fire this from abort()
  recognition().fireEnd();

  assert.equal(handle.getState(), "idle");
  assert.equal(buttonEl.disabled, false);
  assert.match(buttonEl.textContent, /Record/);
  assert.equal(buttonEl.classList.contains("recording"), false);
  assert.equal(statusEl.textContent, "", "an expected self-triggered abort should not show as an error");

  // and it can be started again — not permanently bricked
  buttonEl.click();
  assert.equal(recognition().startCallCount, 2);
});

test("BUG FIX: rapid double-click while starting/stopping is ignored, so start()/abort() can't race", () => {
  const { buttonEl, recognition } = setup();

  buttonEl.click(); // starting
  buttonEl.click(); // should be ignored — button is disabled during "starting"
  assert.equal(recognition().startCallCount, 1);

  recognition().fireStart(); // now listening
  buttonEl.click(); // stopping
  buttonEl.click(); // should be ignored — button is disabled during "stopping"
  assert.equal(recognition().abortCallCount, 1);
});

test("BUG FIX: a synchronous start() failure resets to idle instead of permanently stranding the button", () => {
  const { buttonEl, statusEl, handle, recognition } = setup({ throwOnStart: true });

  buttonEl.click();
  assert.equal(recognition().startCallCount, 1);
  assert.equal(handle.getState(), "idle", "must not stay stuck in 'starting' after a thrown error");
  assert.equal(buttonEl.disabled, false);
  assert.equal(statusEl.textContent, "Could not start voice input.");
});

test("final speech results accumulate in the textarea; interim results are shown live but not committed yet", () => {
  const { buttonEl, textareaEl, recognition } = setup();

  buttonEl.click();
  recognition().fireStart();

  recognition().fireResult([result("hello there", false)], 0);
  assert.equal(textareaEl.value, "hello there", "interim text should be visible immediately");

  recognition().fireResult([result("hello there world", true)], 0);
  assert.equal(textareaEl.value, "hello there world");

  // a second utterance appends onto the first, rather than overwriting it
  recognition().fireResult([result("hello there world", true), result("second thing", false)], 1);
  assert.equal(textareaEl.value, "hello there world second thing");
});

test("BUG FIX: transcribed text is still in the textarea after stopping, so the form actually sends it", () => {
  const { buttonEl, textareaEl, recognition } = setup();

  buttonEl.click();
  recognition().fireStart();
  recognition().fireResult([result("I felt pretty good about it", true)], 0);

  buttonEl.click(); // stop
  recognition().fireError("aborted");
  recognition().fireEnd();

  assert.equal(textareaEl.value, "I felt pretty good about it");
});

test("a real error (e.g. permission denied) resets to idle and explains what happened", () => {
  const { buttonEl, statusEl, handle, recognition } = setup();

  buttonEl.click();
  recognition().fireStart();
  recognition().fireError("not-allowed");

  assert.equal(handle.getState(), "idle");
  assert.match(statusEl.textContent, /Microphone access was denied/);

  // end firing afterwards (as real browsers do) must not throw or change anything unexpectedly
  recognition().fireEnd();
  assert.equal(handle.getState(), "idle");
});

test("handle.stop() forces a stop from outside (e.g. when the surrounding form submits)", () => {
  const { buttonEl, handle, recognition } = setup();

  buttonEl.click();
  recognition().fireStart();

  handle.stop();
  assert.equal(recognition().abortCallCount, 1);
});

test("handle.stop() is a no-op when already idle", () => {
  const { handle, recognition } = setup();
  handle.stop();
  assert.equal(recognition().abortCallCount, 0);
});
