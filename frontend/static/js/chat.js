let conversationId = crypto.randomUUID();
let firstMessage = true;

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

sendBtn.addEventListener("click", sendMessage);
resetBtn.addEventListener("click", resetChat);


function activateChat() {
  if (firstMessage) {
    document.getElementById("welcome-screen").classList.add("hidden");
    chatContainer.classList.add("active");
    resetBtn.classList.remove("hidden");
    firstMessage = false;
  }
}


function appendMessage(role, content) {
  activateChat();

  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "Tú:" : "Tiara:";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = content;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  chatContainer.appendChild(wrap);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  return bubble;
}


// ── Typewriter engine ────────────────────────────────────────────────────────

function createTypewriter(bubble, scrollEl) {
  let queue = [];
  let running = false;
  let currentEl = null;

  function getTextEl() {
    if (!currentEl) {
      currentEl = document.createElement("p");
      currentEl.className = "stream-text";
      bubble.appendChild(currentEl);
    }
    return currentEl;
  }

  function flush() {
    for (const item of queue) {
      item.el.textContent = item.text;
    }
    queue = [];
    running = false;
  }

  function drain() {
    if (queue.length === 0) { running = false; return; }
    running = true;
    const item = queue[0];
    const remaining = item.text.length - item.pos;

    // Faster for longer texts so it doesn't drag
    const batch = remaining > 300 ? 6 : remaining > 100 ? 3 : 1;
    const speed = remaining > 200 ? 10 : 18;

    for (let i = 0; i < batch && item.pos < item.text.length; i++) {
      item.el.textContent += item.text[item.pos++];
    }
    scrollEl.scrollTop = scrollEl.scrollHeight;

    if (item.pos >= item.text.length) {
      queue.shift();
      if (queue.length > 0) setTimeout(drain, speed);
      else running = false;
    } else {
      setTimeout(drain, speed);
    }
  }

  return {
    type(text) {
      const el = getTextEl();
      queue.push({ el, text, pos: 0 });
      if (!running) drain();
    },

    insertTable(html) {
      flush();
      currentEl = null; // next text goes into a fresh <p> after the table

      // Extract text before/after <table> (cache may send them together)
      const lower = html.toLowerCase();
      const tStart = lower.indexOf('<table');
      const tEnd = lower.lastIndexOf('</table>') + 8;

      const before = tStart > 0 ? html.substring(0, tStart).trim() : '';
      const tableHtml = html.substring(tStart, tEnd);
      const after = tEnd < html.length ? html.substring(tEnd).trim() : '';

      if (before) {
        const el = getTextEl();
        el.textContent = before;
        currentEl = null;
      }

      const wrapper = document.createElement("div");
      wrapper.className = "table-container table-fade-in";
      wrapper.innerHTML = tableHtml;
      bubble.appendChild(wrapper);
      scrollEl.scrollTop = scrollEl.scrollHeight;

      if (after) {
        const el = getTextEl();
        queue.push({ el, text: after, pos: 0 });
        if (!running) drain();
      }
    },

    flush,
    isRunning: () => running,
  };
}


// ── Main send ────────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = "";
  appendMessage("user", text);
  sendBtn.disabled = true;

  const botBubble = appendMessage(
    "bot",
    '<span class="thinking">Tiara está pensando<span class="dots"><span>.</span><span>.</span><span>.</span></span></span>'
  );

  const tw = createTypewriter(botBubble, chatContainer);
  let thinkingRemoved = false;

  function removeThinking() {
    if (!thinkingRemoved) {
      const el = botBubble.querySelector(".thinking");
      if (el) el.remove();
      thinkingRemoved = true;
    }
  }

  try {
    const response = await fetch("/api/tiara/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, conversation_id: conversationId })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");

      for (let i = 0; i < parts.length - 1; i++) {
        const line = parts[i].replace("data: ", "");
        if (!line) continue;

        const data = JSON.parse(line);

        if (data.type === "text") {
          removeThinking();
          tw.type(data.content);
        }

        if (data.type === "table") {
          removeThinking();
          tw.insertTable(data.content);
        }

        if (data.type === "chart") {
          tw.flush();
          renderPlotlyChart(botBubble, data.data);
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        if (data.type === "done") {
          tw.flush();
          sendBtn.disabled = false;
        }

        if (data.type === "error") {
          tw.flush();
          botBubble.innerHTML = "Ocurrió un error: " + data.error;
          sendBtn.disabled = false;
        }
      }

      buffer = parts[parts.length - 1];
    }

  } catch (err) {
    tw.flush();
    botBubble.innerHTML = "Error conectando con Tiara.";
    sendBtn.disabled = false;
  }
}


function renderPlotlyChart(bubble, chartData) {
  const wrapper = document.createElement("div");
  wrapper.className = "chart-container";

  const div = document.createElement("div");
  div.style.width = "100%";
  div.style.height = "420px";

  wrapper.appendChild(div);
  bubble.appendChild(wrapper);

  const chart = echarts.init(div, null, { renderer: "canvas" });
  chart.setOption(chartData);

  window.addEventListener("resize", () => chart.resize());
}


async function resetChat() {
  try {
    await fetch(`/api/tiara/conversations/${conversationId}`, { method: "DELETE" });
  } catch (e) {}

  conversationId = crypto.randomUUID();
  document.getElementById("welcome-screen").classList.remove("hidden");
  chatContainer.innerHTML = "";
  chatContainer.classList.remove("active");
  resetBtn.classList.add("hidden");
  firstMessage = true;
  userInput.focus();
}
