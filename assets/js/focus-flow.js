/*
 * Focus flow: scroll-linked emphasis for content blocks.
 *
 * Each block gets a --fx-focus value (0..1) from its distance to the
 * viewport center; the CSS maps it to a GRAD (grade) gain and an ink tint.
 * Grade changes stroke weight without changing glyph widths, so blocks
 * emphasize and relax in place with zero layout shift. Blocks change as a
 * whole unit (the variable is set on the block, inherited by descendants).
 */
(function () {
  "use strict";

  var UNIT_SELECTOR = [
    ".hero-name",
    ".hero-name-cn",
    ".hero-tagline",
    ".hero-bio",
    ".profile-links",
    ".section-title",
    ".section-subtitle",
    ".project-card",
    ".publication-group-title",
    ".pub-list > li:not(.pub-year-mark)",
    ".competition-list > li",
    ".education-list > li",
    ".experience-list > li",
    ".blog-entry",
    ".post-title",
    ".post-meta",
    ".site-main p",
    ".site-main li",
    ".site-main h1",
    ".site-main h2",
    ".site-main h3",
    ".site-main h4",
    ".site-main h5",
    ".site-main h6",
    ".site-main pre",
    ".site-main blockquote",
    ".site-main table"
  ].join(", ");

  // Full strength within INNER of the half-viewport from center; fades out
  // by OUTER. Smoothstep in between keeps the change continuous.
  var INNER = 0.18;
  var OUTER = 0.8;
  var EPS = 0.006;
  var MARGIN = 160;

  var blocks = [];
  var rafId = 0;
  var enabled = false;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function collect() {
    var nodes = document.querySelectorAll(UNIT_SELECTOR);
    blocks = [];
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.closest(".pub-year-mark")) continue;
      // Keep only the outermost matching unit so a block moves as a whole.
      if (el.parentElement && el.parentElement.closest(UNIT_SELECTOR)) continue;
      el.classList.add("fx-block");
      blocks.push({ el: el, value: -1, next: 0 });
    }
  }

  function schedule() {
    if (!enabled) return;
    if (!rafId) {
      rafId = window.requestAnimationFrame(update);
    }
  }

  function curve(d) {
    var t = (OUTER - d) / (OUTER - INNER);
    if (t > 1) {
      t = 1;
    } else if (t < 0) {
      t = 0;
    }
    return t * t * (3 - 2 * t);
  }

  function update() {
    rafId = 0;
    var doc = document.documentElement;
    var vh = window.innerHeight || doc.clientHeight;
    var centerY = vh * 0.5;
    var half = vh * 0.5;
    var maxScroll = Math.max(0, doc.scrollHeight - vh);
    var y = window.pageYOffset || doc.scrollTop || 0;
    var i;
    var b;
    var f;
    // Read phase: geometry only. Writing styles between rect reads would
    // force a synchronous relayout per write, so all writes wait below.
    for (i = 0; i < blocks.length; i++) {
      b = blocks[i];
      var r = b.el.getBoundingClientRect();
      f = 0;
      if (r.height > 0 && r.bottom > -MARGIN && r.top < vh + MARGIN) {
        var center = (r.top + r.bottom) * 0.5;
        f = curve(Math.abs(center - centerY) / half);
        if (f > 0) {
          // Blocks near the document's ends can never meet the viewport
          // center; normalize by the best focus they can actually reach so
          // they still get the full range.
          var centerDoc = center + y;
          var nearest = Math.min(Math.max(centerDoc, centerY), centerY + maxScroll);
          var best = curve(Math.abs(centerDoc - nearest) / half);
          if (best > 0.02 && best < 1) {
            f = Math.min(1, f / best);
          }
        }
      }
      b.next = f;
    }
    // Write phase: only changed values; the engine restyles once per frame.
    for (i = 0; i < blocks.length; i++) {
      b = blocks[i];
      f = b.next;
      if (Math.abs(f - b.value) > EPS || (f === 0 && b.value !== 0) || (f === 1 && b.value !== 1)) {
        b.value = f;
        b.el.style.setProperty("--fx-focus", f.toFixed(3));
      }
    }
  }

  function start() {
    if (enabled) return;
    enabled = true;
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    schedule();
  }

  function stop() {
    if (!enabled) return;
    enabled = false;
    window.removeEventListener("scroll", schedule);
    window.removeEventListener("resize", schedule);
    if (rafId) {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }
    for (var i = 0; i < blocks.length; i++) {
      blocks[i].value = -1;
      // Removing the variable (rather than setting 0) restores the plain
      // design: unset means "no emphasis and no dimming" in the CSS.
      blocks[i].el.style.removeProperty("--fx-focus");
    }
  }

  function onMotionPreference() {
    if (reduceMotion.matches) {
      stop();
    } else {
      start();
    }
  }

  collect();
  if (!blocks.length) return;

  if (reduceMotion.addEventListener) {
    reduceMotion.addEventListener("change", onMotionPreference);
  } else if (reduceMotion.addListener) {
    reduceMotion.addListener(onMotionPreference);
  }

  // Late reflows (webfont load, images, MathJax) change block geometry
  // without a scroll event; recompute when the document resizes.
  if (window.ResizeObserver) {
    new ResizeObserver(schedule).observe(document.documentElement);
  }
  if (document.fonts && document.fonts.ready && document.fonts.ready.then) {
    document.fonts.ready.then(schedule);
  }
  window.addEventListener("load", schedule);
  window.addEventListener("pageshow", schedule);

  onMotionPreference();
})();
