// ── Menú lateral (sidebar) compartido entre Chat y Admin ─────────────────────
(function () {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebar-overlay");
  const toggleBtn = document.getElementById("menu-toggle");
  const closeBtn = document.getElementById("sidebar-close");
  const logoutBtn = document.getElementById("sidebar-logout");

  if (!sidebar || !toggleBtn) return;

  let lastFocused = null;

  function openSidebar() {
    lastFocused = document.activeElement;
    sidebar.classList.add("open");
    overlay.classList.add("visible");
    sidebar.setAttribute("aria-hidden", "false");
    toggleBtn.setAttribute("aria-expanded", "true");
    document.body.classList.add("sidebar-no-scroll");
    document.addEventListener("keydown", onKeydown);
    const firstLink = sidebar.querySelector(".sidebar-link");
    if (firstLink) firstLink.focus();
  }

  function closeSidebar() {
    sidebar.classList.remove("open");
    overlay.classList.remove("visible");
    sidebar.setAttribute("aria-hidden", "true");
    toggleBtn.setAttribute("aria-expanded", "false");
    document.body.classList.remove("sidebar-no-scroll");
    document.removeEventListener("keydown", onKeydown);
    if (lastFocused) lastFocused.focus();
  }

  function onKeydown(e) {
    if (e.key === "Escape") closeSidebar();
  }

  toggleBtn.addEventListener("click", openSidebar);
  closeBtn?.addEventListener("click", closeSidebar);
  overlay?.addEventListener("click", closeSidebar);

  // Resalta el enlace de la página actual
  const current = window.location.pathname === "/admin" ? "admin" : "chat";
  sidebar.querySelectorAll(".sidebar-link[data-page]").forEach((link) => {
    if (link.dataset.page === current) link.classList.add("active");
  });

  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      // Reutiliza el logout() de la página si existe (admin.js ya lo define)
      if (typeof window.logout === "function") {
        window.logout();
        return;
      }
      localStorage.removeItem("tiara_token");
      window.location.href = "/login";
    });
  }
})();
