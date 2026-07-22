function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("No se pudo cargar " + src));
    document.head.appendChild(s);
  });
}

(async function () {
  const shareId = window.location.pathname.split("/").filter(Boolean).pop();

  const noteEl = document.getElementById("shared-note");
  const contentEl = document.getElementById("shared-content");
  const errorEl = document.getElementById("shared-error");
  const questionEl = document.getElementById("shared-question");
  const answerEl = document.getElementById("shared-answer");

  function showError() {
    noteEl.classList.add("hidden");
    errorEl.classList.remove("hidden");
  }

  if (!shareId) {
    showError();
    return;
  }

  try {
    const resp = await fetch(`/api/tiara/share/${shareId}`);
    if (!resp.ok) {
      showError();
      return;
    }

    const data = await resp.json();
    questionEl.textContent = data.question || "";

    let answerHtml = "";
    if (data.html_content) {
      answerHtml += `<div class="table-container">${data.html_content}</div>`;
    }
    if (data.narrative) {
      answerHtml += `<p class="stream-text">${data.narrative}</p>`;
    }
    answerEl.innerHTML = answerHtml;

    if (data.chart_data) {
      await _loadScript("https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js");
      const chartDiv = document.createElement("div");
      chartDiv.className = "shared-chart";
      answerEl.appendChild(chartDiv);
      const chart = echarts.init(chartDiv, null, { renderer: "canvas" });
      chart.setOption(data.chart_data);
      window.addEventListener("resize", () => chart.resize());
    }

    contentEl.classList.remove("hidden");
  } catch (e) {
    showError();
  }
})();
