// ── Auth guard ──────────────────────────────────────────────────────────────
const TOKEN = localStorage.getItem("tiara_token");
if (!TOKEN) window.location.href = "/login";

function authHeaders() {
  return { "Authorization": `Bearer ${TOKEN}`, "Content-Type": "application/json" };
}

async function apiFetch(url, opts = {}) {
  const resp = await fetch(url, { ...opts, headers: { ...authHeaders(), ...(opts.headers || {}) } });
  if (resp.status === 401) { logout(); return null; }
  return resp;
}

function logout() {
  localStorage.removeItem("tiara_token");
  window.location.href = "/login";
}

// ── Toast ────────────────────────────────────────────────────────────────────
function showToast(msg, type = "success") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast show ${type}`;
  setTimeout(() => { el.className = "toast"; }, 3000);
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(s => s.classList.add("hidden"));
  event.target.classList.add("active");
  document.getElementById(`tab-${name}`).classList.remove("hidden");
}

// ── Cache SQL ─────────────────────────────────────────────────────────────────
let cacheEntries = [];

async function loadCache() {
  const resp = await apiFetch("/api/admin/sql-cache");
  if (!resp) return;
  const data = await resp.json();
  cacheEntries = data.entries || [];
  document.getElementById("cache-stats").textContent =
    `${cacheEntries.length} entradas en cache`;
  renderCache(cacheEntries);
}

function renderCache(entries) {
  const tbody = document.getElementById("cache-body");
  if (!entries.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="loading">Sin entradas en cache.</td></tr>';
    return;
  }
  tbody.innerHTML = entries.map(e => `
    <tr>
      <td>${escHtml(e.question)}</td>
      <td class="sql-cell">${escHtml(e.sql)}</td>
      <td>
        <button class="btn-delete" onclick="deleteCache('${e.id}')">Eliminar</button>
      </td>
    </tr>
  `).join("");
}

function filterCache() {
  const q = document.getElementById("cache-search").value.toLowerCase();
  const filtered = q
    ? cacheEntries.filter(e =>
        e.question.toLowerCase().includes(q) || e.sql.toLowerCase().includes(q))
    : cacheEntries;
  renderCache(filtered);
}

async function deleteCache(id) {
  if (!confirm("¿Eliminar esta entrada del cache?")) return;
  const resp = await apiFetch(`/api/admin/sql-cache/${id}`, { method: "DELETE" });
  if (!resp) return;
  if (resp.ok) {
    showToast("Entrada eliminada del cache");
    loadCache();
  } else {
    showToast("Error al eliminar", "error");
  }
}

function showAddForm()  { document.getElementById("add-form").classList.remove("hidden"); }
function hideAddForm()  { document.getElementById("add-form").classList.add("hidden"); }

async function submitCorrection() {
  const question = document.getElementById("new-question").value.trim();
  const sql      = document.getElementById("new-sql").value.trim();
  if (!question || !sql) { showToast("Completa ambos campos", "error"); return; }

  const resp = await apiFetch("/api/admin/sql-cache", {
    method: "POST",
    body: JSON.stringify({ question, sql }),
  });
  if (!resp) return;
  if (resp.ok) {
    showToast("Corrección guardada");
    document.getElementById("new-question").value = "";
    document.getElementById("new-sql").value = "";
    hideAddForm();
    loadCache();
  } else {
    showToast("Error al guardar", "error");
  }
}

// ── Schema Store ──────────────────────────────────────────────────────────────
let schemaEntries = [];

async function loadSchema() {
  const resp = await apiFetch("/api/admin/schema-store");
  if (!resp) return;
  const data = await resp.json();
  schemaEntries = data.entries || [];
  document.getElementById("schema-stats").textContent =
    `${schemaEntries.length} documentos en el schema store`;
  renderSchema(schemaEntries);
}

function renderSchema(entries) {
  const tbody = document.getElementById("schema-body");
  if (!entries.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="loading">Sin documentos en schema store.</td></tr>';
    return;
  }
  tbody.innerHTML = entries.map(e => `
    <tr>
      <td>${escHtml(e.meta?.table || "—")}</td>
      <td>${escHtml(e.meta?.type || "—")}</td>
      <td class="doc-cell">${escHtml((e.doc || "").substring(0, 300))}${e.doc?.length > 300 ? "…" : ""}</td>
      <td>
        <button class="btn-delete" onclick="deleteSchema('${e.id}')">Eliminar</button>
      </td>
    </tr>
  `).join("");
}

function filterSchema() {
  const q = document.getElementById("schema-search").value.toLowerCase();
  const filtered = q
    ? schemaEntries.filter(e =>
        (e.meta?.table || "").toLowerCase().includes(q) ||
        (e.doc || "").toLowerCase().includes(q))
    : schemaEntries;
  renderSchema(filtered);
}

async function deleteSchema(id) {
  if (!confirm("¿Eliminar este documento del schema store?")) return;
  const resp = await apiFetch(`/api/admin/schema-store/${id}`, { method: "DELETE" });
  if (!resp) return;
  if (resp.ok) {
    showToast("Documento eliminado");
    loadSchema();
  } else {
    showToast("Error al eliminar", "error");
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadCache();
loadSchema();
