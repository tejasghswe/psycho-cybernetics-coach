/**
 * Voice input via the browser's Web Speech API, wired as an explicit state
 * machine rather than a single boolean flag.
 *
 * Two real bugs this fixes (see voice-input.test.js for the regression
 * tests that pin the fix):
 *
 * 1. "No way to stop it": recognition.stop() is documented to hang on some
 *    browsers/OSes with continuous=true — it waits to finalize whatever
 *    speech is already in flight before firing `end`, which can take a long
 *    time or never happen. abort() stops immediately and always fires
 *    `end` (after an `error: "aborted"` we treat as expected, not a
 *    failure). A plain boolean `listening` flag can also desync from the
 *    actual recognition object (e.g. start() throws synchronously because
 *    the browser hadn't caught up yet), permanently stranding the button —
 *    the explicit idle/starting/listening/stopping states, with the button
 *    disabled during the two transitional states, make that race
 *    impossible: a click is only ever acted on from a stable state.
 * 2. "Doesn't get saved and sent": a direct consequence of #1 — if the
 *    button can get stuck, the user's stop click is silently dropped, so
 *    whatever they intended to review/edit/send in the textarea never
 *    reliably reflects what they said. Fixing the state machine fixes this
 *    too; there is no separate "save" step — the transcript is written
 *    directly into textareaEl.value as results arrive, which is what the
 *    surrounding form reads at submit time.
 *
 * Dependency-injected SpeechRecognitionCtor (rather than reading
 * window.SpeechRecognition directly) so this is unit-testable with a fake
 * implementation, with no real browser/microphone involved.
 */
function createVoiceInput({ SpeechRecognitionCtor, buttonEl, textareaEl, statusEl }) {
  if (!SpeechRecognitionCtor) {
    buttonEl.hidden = true;
    statusEl.textContent = "Voice input isn't supported in this browser — try Chrome or Edge, or just type.";
    return { stop: () => {}, getState: () => "unsupported" };
  }

  const recognition = new SpeechRecognitionCtor();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  let state = "idle"; // idle | starting | listening | stopping
  let baseText = "";

  function setState(next) {
    state = next;
    if (state === "starting") {
      buttonEl.disabled = true;
      statusEl.textContent = "Starting...";
    } else if (state === "listening") {
      buttonEl.disabled = false;
      buttonEl.textContent = "⏹ Stop";
      buttonEl.classList.add("recording");
      statusEl.textContent = "Listening...";
    } else if (state === "stopping") {
      buttonEl.disabled = true;
      statusEl.textContent = "Stopping...";
    } else {
      buttonEl.disabled = false;
      buttonEl.textContent = "🎤 Record";
      buttonEl.classList.remove("recording");
      statusEl.textContent = "";
    }
  }

  recognition.addEventListener("start", () => {
    setState("listening");
  });

  recognition.addEventListener("result", (event) => {
    let finalChunk = "";
    let interimChunk = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalChunk += transcript;
      } else {
        interimChunk += transcript;
      }
    }
    if (finalChunk) {
      baseText = (baseText ? baseText + " " : "") + finalChunk.trim();
    }
    textareaEl.value = (baseText + (interimChunk ? " " + interimChunk : "")).trim();
  });

  recognition.addEventListener("error", (event) => {
    if (event.error === "aborted") {
      // Expected: we called abort() ourselves to stop recording. The
      // `end` event that follows resets the UI; nothing to report here.
      return;
    }
    // A hard error means the recognition object is done, even on browsers
    // where `end` is slow to follow — don't leave the button stuck. Reset
    // state FIRST: setState("idle") itself clears statusEl.textContent, so
    // the explanatory message below has to be set after, not before.
    setState("idle");
    statusEl.textContent = event.error === "not-allowed"
      ? "Microphone access was denied — check your browser's site permissions."
      : event.error === "no-speech"
      ? "No speech detected."
      : `Voice input error: ${event.error}`;
  });

  recognition.addEventListener("end", () => {
    setState("idle");
  });

  buttonEl.addEventListener("click", () => {
    if (state === "idle") {
      baseText = textareaEl.value;
      setState("starting");
      try {
        recognition.start();
      } catch {
        // Same ordering note as the error handler above: setState("idle")
        // clears the status text, so it must run before this message.
        setState("idle");
        statusEl.textContent = "Could not start voice input.";
      }
    } else if (state === "listening") {
      setState("stopping");
      recognition.abort();
    }
    // "starting"/"stopping": button is disabled, so a click here shouldn't
    // reach this handler in a real browser — ignored defensively anyway.
  });

  return {
    stop: () => {
      if (state === "listening" || state === "starting") {
        recognition.abort();
      }
    },
    getState: () => state,
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { createVoiceInput };
} else {
  (typeof window !== "undefined" ? window : globalThis).createVoiceInput = createVoiceInput;
}
