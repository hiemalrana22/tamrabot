const chatWindow = document.getElementById('chat-window');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');

// Keep session id in localStorage
let sessionId = localStorage.getItem('tamrabot_session_id') || null;

// Helper to generate a new session ID if needed
function generateSessionId() {
  return 'sess-' + Math.random().toString(36).substr(2, 16) + '-' + Date.now();
}

function ensureSessionId() {
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem('tamrabot_session_id', sessionId);
  }
}

function appendMessage(text, sender) {
  const row = document.createElement('div');
  row.className = 'message-row message ' + sender;

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + sender;
  avatar.textContent = sender === 'user' ? '🧑' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  if (sender === 'user') {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Health check on page load
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('http://localhost:9000/health');
    if (!res.ok) throw new Error('Backend not healthy');
    const data = await res.json();
    if (data.status !== 'ok') throw new Error('Backend health check failed');
  } catch (err) {
    appendMessage('Warning: Backend server is not reachable. Please start the backend and reload the page.', 'bot');
  }
});

// After receiving a response from the backend, check for session_id
function handleBotResponse(data) {
  if (!data.session_id) {
    appendMessage('Warning: Session lost or server error. Please refresh or start a new chat.', 'bot');
    // Optionally: generate a new session_id and retry, or reset UI
    // return; // Do not crash, just warn
  } else {
    sessionId = data.session_id;
    localStorage.setItem('tamrabot_session_id', sessionId);
  }
  if (data.response) {
    appendMessage(data.response, 'bot');
  } else if (data.error) {
    appendMessage('Server error: ' + data.error, 'bot');
  } else {
    appendMessage('Unknown server response.', 'bot');
  }
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;
  appendMessage(message, 'user');
  userInput.value = '';
  ensureSessionId();
  try {
    const res = await fetch('http://localhost:9000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId })
    });
    let data = null;
    try {
      data = await res.json();
    } catch (jsonErr) {
      appendMessage('Server error: Invalid or non-JSON response from backend.', 'bot');
      return;
    }
    if (!data || typeof data !== 'object') {
      appendMessage('Server error: Malformed response from backend.', 'bot');
      return;
    }
    if (data.session_id && data.session_id !== sessionId) {
      sessionId = data.session_id;
      localStorage.setItem('tamrabot_session_id', sessionId);
    }
    handleBotResponse(data);
  } catch (err) {
    appendMessage('Error contacting server: ' + (err.message || err), 'bot');
  }
}); 