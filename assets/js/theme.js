(function () {
  "use strict";

  var storageKey = "color-theme";
  var root = document.documentElement;
  var select = document.getElementById("theme-select");

  if (!select) {
    return;
  }

  function normalizeTheme(theme) {
    return theme === "light" || theme === "dark" || theme === "system" ? theme : "system";
  }

  function applyTheme(theme, persist) {
    var normalizedTheme = normalizeTheme(theme);
    root.setAttribute("data-theme", normalizedTheme);
    select.value = normalizedTheme;

    if (persist) {
      try {
        window.localStorage.setItem(storageKey, normalizedTheme);
      } catch (error) {
        // The visual preference still applies for the current page.
      }
    }
  }

  applyTheme(root.getAttribute("data-theme"), false);

  select.addEventListener("change", function (event) {
    applyTheme(event.target.value, true);
  });

  window.addEventListener("storage", function (event) {
    if (event.key === storageKey) {
      applyTheme(event.newValue, false);
    }
  });
})();
