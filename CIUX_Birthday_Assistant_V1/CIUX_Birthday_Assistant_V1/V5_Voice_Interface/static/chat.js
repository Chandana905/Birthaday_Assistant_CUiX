const chat = document.getElementById('chat');
const form = document.getElementById('chat-form');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const dialogueState = document.getElementById('dialogue-state');
const resetButton = document.getElementById('reset-conversation');
const speakButton = document.getElementById('speak-button');
const speechStatus = document.getElementById('speech-status');
const speechError = document.getElementById('speech-error');
const readAloud = document.getElementById('read-aloud');
const monthNames = [
  '', 'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];
const SpeechRecognitionAPI =
  window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;
let startPending = false;
let recognitionError = false;
let stopRequested = false;
let recordingCancelled = false;
let submissionStarted = false;
let finalTranscriptSegments = new Map();
let previousQuestion = '';

function formatDebug(debug) {
  const entities = debug.entities || {};
  const labels = {
    month: 'Month',
    relative_date: 'Relative date',
    relative_month: 'Relative month',
    relationship: 'Relationship',
    person_name: 'Person',
    person_id: 'Person ID'
  };
  const lines = [`Intent: ${debug.intent}`, 'Entities:'];
  const entries = Object.entries(entities);
  if (!entries.length) lines.push('- None');
  for (const [key, value] of entries) {
    const displayValue = key === 'month' ? `${monthNames[value] || value} (${value})` : value;
    lines.push(`- ${labels[key] || key}: ${displayValue}`);
  }
  if (debug.nlu_method) lines.push(`NLU method: ${debug.nlu_method}`);
  if (debug.grounding_status) lines.push(`Grounding: ${debug.grounding_status}`);
  if (debug.fallback_reason) lines.push(`Fallback: ${debug.fallback_reason}`);
  if (debug.ambiguity && debug.ambiguity.includes('person')) {
    lines.push('- Person match: ambiguous; no person was selected');
  }
  return lines.join('\n');
}

function formatDialogueState(state) {
  const fields = [
    ['Previous intent', state.previous_intent || state.last_intent || 'None'],
    ['Remembered month', state.month ? `${monthNames[state.month] || state.month} (${state.month})` : 'None'],
    ['Remembered relationship', state.relationship || 'None'],
    ['Remembered place', state.place || 'None'],
    ['Person', state.person_name || state.person_id || 'None'],
    ['Previous Person ID', state.previous_person_id || 'None'],
    ['Last result IDs', (state.last_result_ids || []).join(', ') || 'None'],
    ['Base result IDs', (state.base_result_ids || []).join(', ') || 'None'],
    ['Clarification pending', state.pending_clarification ? 'Yes' : 'No']
  ];
  return fields.map(([label, value]) => `${label}: ${value}`).join('\n');
}

function addMessage(text, role) {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function normalizeTranscript(text) {
  return text.replace(/\s+/g, ' ').replace(/\s+([,.;!?])/g, '$1').trim();
}

function speakResponse(text) {
  if (!('speechSynthesis' in window) || typeof window.SpeechSynthesisUtterance !== 'function') {
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new window.SpeechSynthesisUtterance(text);
  const voices = window.speechSynthesis.getVoices();
  utterance.voice =
    voices.find(voice => voice.lang.toLowerCase() === 'en-in') ||
    voices.find(voice => voice.lang.toLowerCase().startsWith('en-')) ||
    null;
  utterance.lang = utterance.voice ? utterance.voice.lang : 'en-IN';
  utterance.rate = 1;
  utterance.pitch = 1.02;
  window.speechSynthesis.speak(utterance);
}

async function sendMsg(text = question.value) {
  if (recognition) {
    speechStatus.textContent = 'Stop Speaking before sending your question.';
    return;
  }
  text = text.trim();
  if (!text) return;
  addMessage(text, 'user');
  question.value = '';
  question.disabled = true;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const answer = data.answer || 'No answer returned.';
    addMessage(answer, 'bot');
    if (readAloud.checked) speakResponse(answer);
    trace.textContent = data.debug ? formatDebug(data.debug) : (data.trace || []).join('\n');
    if (data.dialogue_state) dialogueState.textContent = formatDialogueState(data.dialogue_state);
  } catch (error) {
    addMessage(`Error: ${error.message}`, 'bot');
    trace.textContent = `Request failed: ${error.message}`;
  } finally {
    question.disabled = false;
    question.focus();
  }
}

function showSpeechError(message) {
  speechError.textContent = message;
  speechError.hidden = false;
}

function clearSpeechError() {
  speechError.textContent = '';
  speechError.hidden = true;
}

function finishListening(status = '') {
  isListening = false;
  startPending = false;
  speakButton.disabled = false;
  speakButton.textContent = '🎤 Speak';
  speakButton.classList.remove('listening');
  if (status) speechStatus.textContent = status;
}

function cancelListening() {
  if (!recognition) return;
  recordingCancelled = true;
  stopRequested = false;
  speakButton.disabled = true;
  speechStatus.textContent = 'Cancelling recording...';
  try {
    recognition.abort();
  } catch (error) {
    showSpeechError(`Could not cancel speech recognition (${error.name || 'unknown error'}). The transcript will not be sent.`);
    try {
      recognition.stop();
    } catch (stopError) {
      showSpeechError(`Could not cancel or stop speech recognition (${stopError.name || 'unknown error'}). The transcript will not be sent.`);
    }
  }
}

function listen() {
  clearSpeechError();
  if (!SpeechRecognitionAPI) {
    speechStatus.textContent = 'Voice input is not supported by this browser. You can still type your question.';
    speakButton.disabled = true;
    return;
  }
  if (recognition) {
    if (isListening) {
      stopRequested = true;
      speakButton.disabled = true;
      speakButton.textContent = 'Stopping...';
      speechStatus.textContent = 'Stopping recording...';
      try {
        recognition.stop();
      } catch (error) {
        recognitionError = true;
        showSpeechError(`Could not stop speech recognition (${error.name || 'unknown error'}). The transcript will not be sent.`);
      }
    } else {
      speechStatus.textContent = startPending
        ? 'Starting microphone...'
        : 'Finishing the previous voice session...';
    }
    return;
  }

  const activeRecognition = new SpeechRecognitionAPI();
  recognition = activeRecognition;
  startPending = true;
  recognitionError = false;
  stopRequested = false;
  recordingCancelled = false;
  submissionStarted = false;
  finalTranscriptSegments = new Map();
  previousQuestion = question.value;
  speakButton.disabled = true;
  speakButton.textContent = 'Starting...';
  activeRecognition.lang = 'en-IN';
  activeRecognition.interimResults = true;
  activeRecognition.continuous = true;
  activeRecognition.onstart = () => {
    if (recognition !== activeRecognition) return;
    if (recordingCancelled) return;
    isListening = true;
    startPending = false;
    speakButton.disabled = false;
    speakButton.textContent = 'Stop Speaking';
    speakButton.classList.add('listening');
    speechStatus.textContent = 'Listening... Speak your complete question';
  };
  activeRecognition.onresult = event => {
    if (recognition !== activeRecognition) return;
    const interimSegments = [];
    Array.from(event.results || []).forEach((result, index) => {
      const transcript = result[0]?.transcript || '';
      if (result.isFinal) {
        finalTranscriptSegments.set(index, transcript);
      } else if (transcript) {
        interimSegments.push(transcript);
      }
    });
    const finalTranscript = Array.from(finalTranscriptSegments.entries())
      .sort(([left], [right]) => left - right)
      .map(([, transcript]) => transcript)
      .join(' ');
    question.value = normalizeTranscript([finalTranscript, ...interimSegments].join(' '));
  };
  activeRecognition.onerror = event => {
    if (recognition !== activeRecognition) return;
    const messages = {
      'not-allowed': 'Microphone access was denied or blocked. Allow microphone access for this site in Chrome settings, then try again.',
      'service-not-allowed': 'The browser blocked its speech recognition service. Check site permissions or browser policy, then try again.',
      'audio-capture': 'No microphone was found. Connect or enable a microphone, then try again.',
      'no-speech': 'No speech detected. Please try again.',
      network: "Chrome's online speech recognition service could not be reached. This is separate from the chatbot server. Check your internet connection and any VPN, firewall, or browser policy; you can also type your question.",
      'language-not-supported': 'The browser speech service does not support en-IN. You can still type your question.',
      aborted: 'Speech recognition was interrupted. Your transcript was not sent.'
    };
    startPending = false;
    recognitionError = true;
    if (!(recordingCancelled && event.error === 'aborted')) {
      showSpeechError(messages[event.error] || `Speech recognition failed (${event.error || 'unknown error'}). Your transcript was not sent.`);
      speechStatus.textContent = 'Voice input interrupted.';
    }
    speakButton.disabled = true;
  };
  activeRecognition.onend = () => {
    if (recognition !== activeRecognition) return;
    const finalTranscript = normalizeTranscript(
      Array.from(finalTranscriptSegments.entries())
        .sort(([left], [right]) => left - right)
        .map(([, transcript]) => transcript)
        .join(' ')
    );
    const shouldSubmit = stopRequested && !recordingCancelled && !recognitionError;
    const wasCancelled = recordingCancelled;
    const endedUnexpectedly = !stopRequested && !wasCancelled && !recognitionError;
    recognition = null;
    stopRequested = false;
    recordingCancelled = false;
    finalTranscriptSegments.clear();
    if (wasCancelled) {
      question.value = previousQuestion;
      finishListening('Recording cancelled. Nothing was sent.');
    } else if (recognitionError) {
      question.value = finalTranscript || previousQuestion;
      finishListening('Voice input ended without sending.');
    } else if (shouldSubmit) {
      if (!finalTranscript) {
        question.value = previousQuestion;
        showSpeechError('No speech detected. Please try again.');
        finishListening('Recording ended.');
      } else {
        question.value = finalTranscript;
        finishListening('Recording complete. Sending your question...');
        if (!submissionStarted) {
          submissionStarted = true;
          void sendMsg(finalTranscript);
        }
      }
    } else if (endedUnexpectedly) {
      question.value = finalTranscript || previousQuestion;
      showSpeechError('Speech recognition ended before you stopped it. The transcript was not sent; please try again or type your question.');
      finishListening('Voice input ended without sending.');
    }
  };

  try {
    activeRecognition.start();
  } catch (error) {
    if (recognition === activeRecognition) {
      recognition = null;
      recognitionError = true;
      const message = error.name === 'NotAllowedError' || error.name === 'SecurityError'
        ? 'Microphone access was denied or blocked. Allow microphone access for this site in Chrome settings, then try again.'
        : `Could not start browser speech recognition (${error.name || 'unknown error'}). Check microphone access and try again, or type your question.`;
      question.value = previousQuestion;
      showSpeechError(message);
      finishListening('Voice input ended.');
    }
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  sendMsg();
});

document.querySelectorAll('.chip').forEach(button => {
  button.addEventListener('click', () => sendMsg(button.dataset.q));
});

speakButton.addEventListener('click', listen);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && recognition) {
    event.preventDefault();
    cancelListening();
  }
});
if (!SpeechRecognitionAPI) {
  speakButton.disabled = true;
  speechStatus.textContent = 'Voice input is not supported by this browser. You can still type your question.';
} else {
  speechStatus.textContent = 'Voice input is ready. Microphone access requires localhost or HTTPS.';
}

if (!('speechSynthesis' in window) || typeof window.SpeechSynthesisUtterance !== 'function') {
  readAloud.disabled = true;
  readAloud.title = 'Speech output is not supported by this browser.';
}
readAloud.addEventListener('change', () => {
  if (!readAloud.checked && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
});

resetButton.addEventListener('click', async () => {
  resetButton.disabled = true;
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  cancelListening();
  try {
    const response = await fetch('/reset', {method: 'POST'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!data.success) throw new Error('The server did not reset the conversation.');
    chat.replaceChildren();
    addMessage(data.answer || 'Conversation reset.', 'bot');
    trace.textContent = 'Conversation reset.';
    dialogueState.textContent = formatDialogueState(data.dialogue_state || {});
  } catch (error) {
    addMessage(`Reset failed: ${error.message}`, 'bot');
  } finally {
    resetButton.disabled = false;
  }
});

addMessage('Hello! I am Version 5, your voice-enabled birthday assistant. You can type or speak your questions; birthday facts are grounded in the workbook.', 'bot');
dialogueState.textContent = formatDialogueState(window.initialDialogueState || {});
