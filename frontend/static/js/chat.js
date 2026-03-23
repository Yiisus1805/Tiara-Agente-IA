let conversationId = 'session-' + Date.now();
let useStreaming = true;

function addMessage(role, content, toolCalls = null) {
  const messagesDiv = document.getElementById('messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;

  const label = role === 'user' ? 'Tú' : 'TIARA';
  let html = `<div class="message-label">${label}:</div><div class="message-content">`;

  let formattedContent = content
    .replace(/```sql\n([\s\S]+?)```/g, '<pre><code>$1</code></pre>')
    .replace(/```([\s\S]+?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/<table/g, '<div class="table-wrapper"><table')
    .replace(/<\/table>/g, '</table></div>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/```sql\n([\s\S]+?)```/g, '<pre><code>$1</code></pre>')
    .replace(/```([\s\S]+?)```/g, '<pre><code>$1</code></pre>')
    .replace(/\n/g, '<br>');

  html += formattedContent;

  if (toolCalls && toolCalls.length > 0) {
    toolCalls.forEach(tc => {
      html += `<div class="tool-call">
        🔧 Tool: <strong>${tc.name}</strong>
        ${tc.tool_args ? '<br>Args: ' + JSON.stringify(tc.tool_args) : ''}
      </div>`;
    });
  }

  html += '</div>';
  messageDiv.innerHTML = html;
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showLoading() {
  const messagesDiv = document.getElementById('messages');

  const messageDiv = document.createElement('div');
  messageDiv.className = 'message assistant';
  messageDiv.id = 'loading';

  const labelDiv = document.createElement('div');
  labelDiv.className = 'message-label';
  labelDiv.textContent = 'TIARA:';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content loading-bubble';
  contentDiv.innerHTML = `
    <span class="thinking-dots">
      TIARA está pensando
      <span class="dot">.</span>
      <span class="dot">.</span>
      <span class="dot">.</span>
    </span>
  `;

  messageDiv.appendChild(labelDiv);
  messageDiv.appendChild(contentDiv);
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function hideLoading() {
  const loading = document.getElementById('loading');
  if (loading) loading.remove();
}

function showError(message) {
  const messagesDiv = document.getElementById('messages');
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error';
  errorDiv.textContent = '❌ Error: ' + message;
  messagesDiv.appendChild(errorDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessageNormal(question) {
  try {
    const response = await fetch('/api/tiara/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, conversation_id: conversationId })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Error en la respuesta');
    }

    hideLoading();
    addMessage('assistant', data.answer, data.tool_calls);
  } catch (error) {
    hideLoading();
    showError(error.message);
  }
}

async function sendMessageStreaming(question) {
  try {
    const response = await fetch('/api/tiara/chat_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, conversation_id: conversationId })
    });

    if (!response.ok) throw new Error('Error en la respuesta');

    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const labelDiv = document.createElement('div');
    labelDiv.className = 'message-label';
    labelDiv.textContent = 'TIARA:';
    messageDiv.appendChild(labelDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    messageDiv.appendChild(contentDiv);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let firstChunk = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        const data = JSON.parse(line.slice(6));

        if (data.type === 'content') {
          if (firstChunk) {
            hideLoading();
            messagesDiv.appendChild(messageDiv);
            firstChunk = false;
          }
          const wrapped = data.content
            .replace(/<table/g, '<div class="table-wrapper"><table')
            .replace(/<\/table>/g, '</table></div>');
          contentDiv.innerHTML += wrapped;
          messagesDiv.scrollTop = messagesDiv.scrollHeight;
        } else if (data.type === 'tool_call') {
          const toolDiv = document.createElement('div');
          toolDiv.className = 'tool-call';
          toolDiv.innerHTML = `🔧 Tool: <strong>${data.tool_name}</strong>`;
          contentDiv.appendChild(toolDiv);
        } else if (data.type === 'error') {
          hideLoading();
          showError(data.error);
        }
      }
    }

    if (firstChunk) hideLoading();

  } catch (error) {
    hideLoading();
    showError(error.message);
  }
}

async function sendMessage() {
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question) return;

  addMessage('user', question);
  input.value = '';

  const sendBtn = document.getElementById('sendBtn');
  sendBtn.disabled = true;
  showLoading();

  if (useStreaming) {
    await sendMessageStreaming(question);
  } else {
    await sendMessageNormal(question);
  }

  sendBtn.disabled = false;
  input.focus();
}

function handleKeyPress(event) {
  if (event.key === 'Enter') sendMessage();
}

const WELCOME_MSG = 'Bienvenido a TIARA  ¿En qué puedo ayudarte hoy?';

function newConversation() {
  conversationId = 'session-' + Date.now();
  document.getElementById('messages').innerHTML = '';
  addMessage('assistant', WELCOME_MSG);
  document.getElementById('questionInput').focus();
}

document.getElementById('questionInput').focus();
addMessage('assistant', WELCOME_MSG);