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


function wrapTable(html) {
  // Envuelve cada <table> en un contenedor con scroll
  return html.replace(/(<table[\s\S]*?<\/table>)/gi, '<div class="table-container">$1</div>');
}

function cleanAfterTable(html) {
  const tableEnd = html.lastIndexOf('</table>');
  if (tableEnd === -1) return html;

  const beforeTable = html.substring(0, tableEnd + '</table>'.length);
  const afterTable = html.substring(tableEnd + '</table>'.length).trim();

  if (!afterTable) return beforeTable;

  // Eliminar líneas que son filas de tabla markdown (| val | val |)
  const lines = afterTable.split('\n');
  const cleanLines = lines.filter(line => {
    const trimmed = line.trim();
    if (/^\|.*\|$/.test(trimmed)) return false;   
    if (/^\|[\s\-|]+\|$/.test(trimmed)) return false; 
    return true;
  });

  const remainingText = cleanLines.join('\n').trim();

  // Si queda texto útil (explicación), conservarlo
  return remainingText ? beforeTable + '\n' + remainingText : beforeTable;
}


async function sendMessage() {

  const text = userInput.value.trim();

  if (!text) return;

  userInput.value = "";

  appendMessage("user", text);

  sendBtn.disabled = true;

  const botBubble = appendMessage(
    "bot",
    'Tiara está pensando<span class="dots"><span>.</span><span>.</span><span>.</span></span>'
  );

  try {

    const response = await fetch("/api/tiara/chat_stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: text,
        conversation_id: conversationId
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let accumulatedContent = "";

    while (true) {

      const { value, done } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");

      for (let i = 0; i < parts.length - 1; i++) {

        const line = parts[i].replace("data: ", "");

        if (!line) continue;

        const data = JSON.parse(line);

        if (data.type === "content") {

          if (botBubble.innerHTML.includes("Tiara está pensando")) {
            botBubble.innerHTML = "";
            accumulatedContent = "";
          }

          accumulatedContent += data.content;

          // Si hay tabla completa, limpiar texto posterior y envolver
          if (/<table[\s\S]*<\/table>/i.test(accumulatedContent)) {
            const cleaned = cleanAfterTable(accumulatedContent);
            botBubble.innerHTML = wrapTable(cleaned);
          } else {
            botBubble.innerHTML = accumulatedContent;
          }

          chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        if (data.type === "done") {
          sendBtn.disabled = false;
        }

        if (data.type === "error") {
          botBubble.innerHTML = " Ocurrió un error: " + data.error;
          sendBtn.disabled = false;
        }

      }

      buffer = parts[parts.length - 1];
    }

  } catch (err) {

    botBubble.innerHTML = " Error conectando con Tiara.";
    sendBtn.disabled = false;
  }

}


async function resetChat() {

  try {

    await fetch(`/api/tiara/conversations/${conversationId}`, {
      method: "DELETE"
    });

  } catch (e) { }

  conversationId = crypto.randomUUID();
  document.getElementById("welcome-screen").classList.remove("hidden");
  chatContainer.innerHTML = "";
  chatContainer.classList.remove("active");

  resetBtn.classList.add("hidden");

  firstMessage = true;

  userInput.focus();
}