const startPanel = document.getElementById("start-panel");
const crisisPanel = document.getElementById("crisis-panel");
const conversationPanel = document.getElementById("conversation-panel");
const reportPanel = document.getElementById("report-panel");

const startForm = document.getElementById("start-form");
const startBtn = document.getElementById("start-btn");
const startError = document.getElementById("start-error");

const startThinking = document.getElementById("start-thinking");

const answerForm = document.getElementById("answer-form");
const answerBtn = document.getElementById("answer-btn");
const answerError = document.getElementById("answer-error");
const answerThinking = document.getElementById("answer-thinking");

let sessionId = null;
let qaHistory = [];

// Voice input logic (state machine, testable) lives in voice-input.js,
// loaded before this file. Not supported in Firefox/Safari — createVoiceInput
// hides the mic button and explains why rather than leaving a dead button.
const _SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

const startVoice = createVoiceInput({
  SpeechRecognitionCtor: _SpeechRecognitionCtor,
  buttonEl: document.getElementById("start-mic-btn"),
  textareaEl: document.getElementById("transcript-input"),
  statusEl: document.getElementById("start-voice-status"),
});
const answerVoice = createVoiceInput({
  SpeechRecognitionCtor: _SpeechRecognitionCtor,
  buttonEl: document.getElementById("answer-mic-btn"),
  textareaEl: document.getElementById("answer-input"),
  statusEl: document.getElementById("answer-voice-status"),
});

function showOnly(panel) {
  for (const p of [startPanel, crisisPanel, conversationPanel, reportPanel]) {
    p.hidden = p !== panel;
  }
}

function resetToStart() {
  startVoice.stop();
  answerVoice.stop();
  sessionId = null;
  qaHistory = [];
  document.getElementById("transcript-input").value = "";
  startError.textContent = "";
  showOnly(startPanel);
}

function renderQaHistory() {
  const container = document.getElementById("qa-history");
  container.innerHTML = "";
  for (const turn of qaHistory) {
    const div = document.createElement("div");
    div.className = "qa-turn";
    const q = document.createElement("div");
    q.className = "q";
    q.textContent = turn.question;
    const a = document.createElement("div");
    a.className = "a";
    a.textContent = turn.answer;
    div.appendChild(q);
    div.appendChild(a);
    container.appendChild(div);
  }
}

function renderReport(report) {
  document.getElementById("report-reframe").textContent = report.self_image_reframe;

  const goodThingsList = document.getElementById("report-good-things");
  goodThingsList.innerHTML = "";
  for (const item of report.good_things) {
    const li = document.createElement("li");
    li.textContent = item;
    goodThingsList.appendChild(li);
  }

  const actionablesList = document.getElementById("report-actionables");
  actionablesList.innerHTML = "";
  for (const item of report.actionables) {
    const li = document.createElement("li");
    li.textContent = item;
    actionablesList.appendChild(li);
  }
}

function renderSession(body) {
  sessionId = body.session_id;
  qaHistory = body.qa_turns || [];

  if (body.status === "crisis_terminated") {
    document.getElementById("crisis-message").textContent = body.crisis_message;
    showOnly(crisisPanel);
    return;
  }

  if (body.status === "completed") {
    renderReport(body.report);
    showOnly(reportPanel);
    return;
  }

  renderQaHistory();
  document.getElementById("current-question").textContent = body.question;
  document.getElementById("answer-input").value = "";
  answerError.textContent = "";
  showOnly(conversationPanel);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "Request failed.";
    throw new Error(detail);
  }
  return body;
}

startForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  startVoice.stop();
  startBtn.disabled = true;
  startThinking.hidden = false;
  startError.textContent = "";
  try {
    const body = await postJson("/sessions", {
      mood_selection: document.getElementById("mood-select").value,
      transcript: document.getElementById("transcript-input").value,
    });
    renderSession(body);
  } catch (err) {
    startError.textContent = err.message;
  } finally {
    startBtn.disabled = false;
    startThinking.hidden = true;
  }
});

answerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  answerVoice.stop();
  answerBtn.disabled = true;
  answerThinking.hidden = false;
  answerError.textContent = "";
  try {
    const body = await postJson(`/sessions/${sessionId}/answer`, {
      answer: document.getElementById("answer-input").value,
    });
    renderSession(body);
  } catch (err) {
    answerError.textContent = err.message;
  } finally {
    answerBtn.disabled = false;
    answerThinking.hidden = true;
  }
});

document.getElementById("crisis-restart-btn").addEventListener("click", resetToStart);
document.getElementById("report-restart-btn").addEventListener("click", resetToStart);
