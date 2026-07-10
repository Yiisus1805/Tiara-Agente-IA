// ── Auth guard ──────────────────────────────────────────────────────────────
const _token = localStorage.getItem("tiara_token");
if (!_token) window.location.href = "/login";

function _authHeaders() {
  return { "Authorization": `Bearer ${_token}`, "Content-Type": "application/json" };
}

function _handleUnauth() {
  localStorage.removeItem("tiara_token");
  window.location.href = "/login";
}

// ── Inactivity timeout (30 min) ──────────────────────────────────────────────
const INACTIVITY_MS = 30 * 60 * 1000;
let _inactivityTimer = null;

function _resetInactivityTimer() {
  clearTimeout(_inactivityTimer);
  _inactivityTimer = setTimeout(() => {
    localStorage.removeItem("tiara_token");
    window.location.href = "/login?reason=inactividad";
  }, INACTIVITY_MS);
}

["mousemove", "keydown", "click", "touchstart"].forEach(evt =>
  document.addEventListener(evt, _resetInactivityTimer, { passive: true })
);
_resetInactivityTimer();

// ── Carga diferida de librerías pesadas (solo cuando se usan) ────────────────
function _loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("No se pudo cargar " + src));
    document.head.appendChild(s);
  });
}

let _echartsPromise = null;
function ensureEcharts() {
  if (typeof echarts !== "undefined") return Promise.resolve();
  if (!_echartsPromise) _echartsPromise = _loadScript("https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js");
  return _echartsPromise;
}

let _xlsxPromise = null;
function ensureXLSX() {
  if (typeof XLSX !== "undefined") return Promise.resolve();
  if (!_xlsxPromise) _xlsxPromise = _loadScript("https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js");
  return _xlsxPromise;
}

let conversationId = crypto.randomUUID();
let firstMessage = true;

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !sendBtn.disabled) sendMessage();
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
      // Solo agregar los caracteres que drain no animó aún
      item.el.textContent += item.text.substring(item.pos);
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

      // Toolbar con botón de exportar Excel
      const outerWrap = document.createElement("div");
      outerWrap.className = "table-outer-wrap table-fade-in";

      const toolbar = document.createElement("div");
      toolbar.className = "chart-toolbar";

      const exportBtn = document.createElement("button");
      exportBtn.className = "chart-export-btn";
      exportBtn.title = "Exportar tabla a Excel";
      exportBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Exportar Excel`;

      toolbar.appendChild(exportBtn);
      outerWrap.appendChild(toolbar);

      const wrapper = document.createElement("div");
      wrapper.className = "table-container";
      wrapper.innerHTML = tableHtml;
      outerWrap.appendChild(wrapper);

      bubble.appendChild(outerWrap);
      scrollEl.scrollTop = scrollEl.scrollHeight;

      exportBtn.addEventListener("click", async () => {
        const tableEl = wrapper.querySelector("table");
        if (!tableEl) return;
        await ensureXLSX();
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.table_to_sheet(tableEl);
        XLSX.utils.book_append_sheet(wb, ws, "Datos");
        XLSX.writeFile(wb, "datos-tiara.xlsx");
      });

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


// ── SSE stream helper ────────────────────────────────────────────────────────

const SSE_IDLE_TIMEOUT_MS = 90000; // 90 s sin datos → reintento automático

function readWithTimeout(reader, ms) {
  return Promise.race([
    reader.read(),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("sse_timeout")), ms)
    ),
  ]);
}

async function streamIntoBubble(question, bubble, isRetry) {
  const tw = createTypewriter(bubble, chatContainer);
  let thinkingRemoved = false;

  // Aviso de consulta lenta — aparece a los 10s si no hay respuesta aún
  const slowTimer = setTimeout(() => {
    if (!thinkingRemoved) {
      const el = bubble.querySelector(".thinking");
      if (el) el.innerHTML =
        'Procesando consulta, esto puede tardar un momento<span class="dots"><span>.</span><span>.</span><span>.</span></span>';
    }
  }, 10000);

  function removeThinking() {
    if (!thinkingRemoved) {
      clearTimeout(slowTimer);
      const el = bubble.querySelector(".thinking");
      if (el) el.remove();
      thinkingRemoved = true;
    }
  }

  function showRetryError(message) {
    tw.flush();
    removeThinking();
    bubble.innerHTML = "";

    const wrap = document.createElement("div");
    wrap.className = "error-retry";

    const msg = document.createElement("p");
    msg.className = "error-msg";
    msg.textContent = message || "No pude generar una respuesta. Puedes intentarlo de nuevo.";

    const btn = document.createElement("button");
    btn.className = "btn retry-btn";
    btn.textContent = "↺ Intentar de nuevo";
    btn.onclick = () => retryMessage(question, bubble);

    wrap.appendChild(msg);
    wrap.appendChild(btn);
    bubble.appendChild(wrap);
    sendBtn.disabled = false;
  }

  try {
    const response = await fetch("/api/tiara/chat_stream", {
      method: "POST",
      headers: _authHeaders(),
      body: JSON.stringify({ question, conversation_id: conversationId, retry: isRetry })
    });

    if (response.status === 401) { _handleUnauth(); return; }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      let value, done;
      try {
        ({ value, done } = await readWithTimeout(reader, SSE_IDLE_TIMEOUT_MS));
      } catch (e) {
        if (e.message === "sse_timeout") {
          showRetryError("La consulta tardó demasiado. Puedes intentarlo de nuevo.");
          return;
        }
        throw e;
      }
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
          await renderPlotlyChart(bubble, data.data);
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        if (data.type === "done") {
          tw.flush();
          sendBtn.disabled = false;
          addShareButton(bubble, question);
        }

        if (data.type === "error_retry") {
          showRetryError(data.message);
        }

        if (data.type === "error") {
          tw.flush();
          bubble.innerHTML = "Ocurrió un error: " + data.error;
          sendBtn.disabled = false;
        }
      }

      buffer = parts[parts.length - 1];
    }

  } catch (err) {
    clearTimeout(slowTimer);
    tw.flush();
    bubble.innerHTML = "No pude conectarme en este momento. Verifica tu conexión e intenta de nuevo.";
    sendBtn.disabled = false;
  }
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

  await streamIntoBubble(text, botBubble, false);
}


// ── Retry (same bubble, cache evictado en backend) ───────────────────────────

async function retryMessage(question, bubble) {
  bubble.innerHTML = '<span class="thinking">Tiara está pensando<span class="dots"><span>.</span><span>.</span><span>.</span></span></span>';
  sendBtn.disabled = true;
  await streamIntoBubble(question, bubble, true);
}


async function renderPlotlyChart(bubble, chartData) {
  await ensureEcharts();

  const wrapper = document.createElement("div");
  wrapper.className = "chart-container";

  const toolbar = document.createElement("div");
  toolbar.className = "chart-toolbar";

  const exportBtn = document.createElement("button");
  exportBtn.className = "chart-export-btn";
  exportBtn.title = "Exportar gráfico como imagen PNG";
  exportBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Exportar`;

  toolbar.appendChild(exportBtn);

  const div = document.createElement("div");
  div.className = "chart-render-target";
  div.style.width = "100%";
  div.style.height = "420px";

  wrapper.appendChild(toolbar);
  wrapper.appendChild(div);
  bubble.appendChild(wrapper);

  const chart = echarts.init(div, null, { renderer: "canvas" });
  chart.setOption(chartData);

  exportBtn.addEventListener("click", () => {
    const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: "#ffffff" });
    const a = document.createElement("a");
    a.href = url;
    a.download = "grafico-tiara.png";
    a.click();
  });

  window.addEventListener("resize", () => chart.resize());
}


// ── Compartir resultado por link ─────────────────────────────────────────────

function addShareButton(bubble, question) {
  if (bubble.querySelector(".share-btn")) return;

  const btn = document.createElement("button");
  btn.className = "share-btn";
  btn.type = "button";
  btn.title = "Compartir este resultado";
  btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.6" y1="10.6" x2="15.4" y2="6.4"/><line x1="8.6" y1="13.4" x2="15.4" y2="17.6"/></svg> Compartir`;
  btn.addEventListener("click", () => shareBubble(bubble, question, btn));
  bubble.appendChild(btn);
}

async function shareBubble(bubble, question, btn) {
  const tableEl = bubble.querySelector("table");
  const html_content = tableEl ? tableEl.outerHTML : "";

  const narrative = Array.from(bubble.querySelectorAll(".stream-text"))
    .map((p) => p.textContent.trim())
    .filter(Boolean)
    .join("\n\n");

  let chart_data = null;
  const chartDiv = bubble.querySelector(".chart-render-target");
  if (chartDiv && typeof echarts !== "undefined") {
    const instance = echarts.getInstanceByDom(chartDiv);
    if (instance) chart_data = instance.getOption();
  }

  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = "Generando link...";

  try {
    const resp = await fetch("/api/tiara/share", {
      method: "POST",
      headers: _authHeaders(),
      body: JSON.stringify({ question, html_content, narrative, chart_data }),
    });

    if (resp.status === 401) { _handleUnauth(); return; }
    if (!resp.ok) throw new Error("share failed");

    const data = await resp.json();
    const fullUrl = window.location.origin + data.url;
    await navigator.clipboard.writeText(fullUrl);
    btn.textContent = "✓ Link copiado (expira en 2h)";
  } catch (e) {
    btn.textContent = "No se pudo generar el link";
  } finally {
    setTimeout(() => {
      btn.innerHTML = originalHtml;
      btn.disabled = false;
    }, 2500);
  }
}


// Detección de conexión

const offlineBanner = document.getElementById("offline-banner");

window.addEventListener("offline", () => {
  offlineBanner.classList.add("visible");
});

window.addEventListener("online", () => {
  offlineBanner.classList.remove("visible");
});


async function resetChat() {
  try {
    await fetch(`/api/tiara/conversations/${conversationId}`, {
      method: "DELETE", headers: _authHeaders(),
    });
  } catch (e) {}

  conversationId = crypto.randomUUID();
  document.getElementById("welcome-screen").classList.remove("hidden");
  chatContainer.innerHTML = "";
  chatContainer.classList.remove("active");
  resetBtn.classList.add("hidden");
  firstMessage = true;
  userInput.focus();
}
