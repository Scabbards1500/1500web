export function mountBlogDecor() {
  const scope = document.querySelector(".blog-site");
  if (!scope) return;
  const before = new Set(scope.querySelectorAll("*"));
  const side = scope.querySelector(".profile-sidebar");
  const sideMarkup = side?.innerHTML;
  const controller = new AbortController();
  let scrollFrame = 0;
  let disposed = false;
  var root = document.documentElement;
  var reduceMotion = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  if (!reduceMotion) root.classList.add("motion");
  try {
    root.dataset.theme = localStorage.getItem("site-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  } catch (error) { root.dataset.theme = "light"; }

  /* ---------------- Masked reveal ---------------- */

  // Checked from the scroll loop with getBoundingClientRect, which (unlike
  // IntersectionObserver) is not affected by the element's own clip-path.
  var pendingReveals = [];
  var requestScrollUpdate = function () {};

  function checkReveals() {
    if (!pendingReveals.length) return;
    var limit = window.innerHeight * 0.92;
    pendingReveals = pendingReveals.filter(function (el) {
      if (!el.isConnected) return false;
      var rect = el.getBoundingClientRect();
      if (rect.height === 0 || rect.top > limit || rect.bottom < 0) return true;
      el.classList.add("in");
      var delay = parseFloat(el.style.getPropertyValue("--rd")) || 0;
      window.setTimeout(function () {
        el.classList.add("done");
      }, delay + 1100);
      return false;
    });
  }

  function reveal(el, delay) {
    if (reduceMotion || !el) return;
    el.classList.add("reveal");
    if (delay) el.style.setProperty("--rd", delay + "ms");
    pendingReveals.push(el);
    requestScrollUpdate();
  }

  /* ---------------- Floating decorations ---------------- */

  var SVG_NS = "http://www.w3.org/2000/svg";
  var ICONS = {
    satellite: {
      viewBox: "0 0 240 160",
      parts: [
        ["sat-link", "M45 80H195"],
        ["sat-panel", "M12 48H79V109H12ZM161 48H228V109H161Z"],
        ["sat-grid", "M34 48V109M57 48V109M12 68H79M12 89H79M183 48V109M206 48V109M161 68H228M161 89H228"],
        ["sat-body", "M100 57H140V101H100Z"],
        ["sat-detail", "M108 65H132V73H108ZM110 81H130M110 88H124M120 101V120M111 120H129"],
        ["sat-dish", "M107 40Q120 62 133 40ZM120 40V29M116 29H124M120 51V57"],
        ["sat-signal", "M105 24Q120 11 135 24M98 16Q120 -2 142 16"],
      ],
    },
    bird: {
      viewBox: "0 0 40 18",
      d: "M1 9.5C6.5 3.5 12 3 20 10.5 28 3 33.5 3.5 39 9.5 33.5 7.6 27 9 20 15.5 13 9 6.5 7.6 1 9.5Z",
    },
    feather: {
      viewBox: "0 0 32 52",
      d: "M22 2C31 14 29 32 12 46M22 2C11 13 7 30 12 46M12 46 7 51M19 9l6 3M16 16l10 3M14 23l11 2M13 30l10 1M12 37l7 0M18 12l-7 4M15 20l-7 5M14 28l-6 5",
    },
    // White gull with spread wings in a three-quarter view: each wing is one
    // clean crescent tapering to a point, near wing up-left, far wing dropping
    // right. Light blueprint outline with a faint dot fill.
    gull: {
      viewBox: "0 0 210 84",
      parts: [
        ["d-body", "M2 16C22 6 52 2 72 10C84 15 92 24 98 33C101 26 106 21 113 20C119 19 124 20 128 22L135 21.5L127 26.5C125 31 125 36 127 40C129 43 132 45 136 46C160 44 188 44 198 50C204 56 207 68 208 82C196 66 166 60 128 61C121 61.5 117 63 116 66L118 78L93 72L97 56C74 42 40 26 2 16Z"],
        ["d-dots", "M2 16C22 6 52 2 72 10C84 15 92 24 98 33C101 26 106 21 113 20C119 19 124 20 128 22L135 21.5L127 26.5C125 31 125 36 127 40C129 43 132 45 136 46C160 44 188 44 198 50C204 56 207 68 208 82C196 66 166 60 128 61C121 61.5 117 63 116 66L118 78L93 72L97 56C74 42 40 26 2 16Z"],
      ],
    },
    // Flat cumulus with a pale shaded base.
    cloud: {
      viewBox: "0 0 160 56",
      parts: [
        ["c-body", "M16 50C6 50 4 39 13 36.5C12 27 24 21 33 26C38 13 56 9 67 19C74 7 97 7 102 21C109 15 123 17 126 28C138 26 151 34 146 44C153 46 151 50 144 50Z"],
        ["c-shade", "M10.5 45C30 47 120 47 148.5 45.6C149.4 48.6 147.5 50 144 50H16C13 50 11.2 47.8 10.5 45Z"],
      ],
    },
    // Stylised wind streaks, a line-art cloud.
    cloudline: {
      viewBox: "0 0 120 30",
      parts: [["c-line", "M22 7H78M8 15H112M34 23H66"]],
    },
  };

  var NIGHT_ICONS = {
    bird: { viewBox: "0 0 40 40", d: "M20 2L24 15L38 20L24 24L20 38L16 24L2 20L16 15Z" },
    gull: { viewBox: "0 0 100 60", d: "M38 30a12 12 0 1 0 24 0a12 12 0 1 0-24 0M5 47C-4 34 91 4 96 16C105 29 9 59 5 47Z" },
    cloud: { viewBox: "0 0 100 60", d: "M14 47Q35 3 89 12M21 54Q43 13 94 22M34 49Q55 27 86 30M67 6L70 15L79 18L70 21L67 30L64 21L55 18L64 15Z" },
    cloudline: { viewBox: "0 0 120 40", d: "M3 35L85 9M25 37L98 14M86 3L89 10L97 12L89 15L86 23L83 15L75 12L83 10Z" },
    feather: { viewBox: "0 0 50 60", d: "M6 49L17 27L34 36L44 8M3 49h6M6 46v6M14 27h6M17 24v6M31 36h6M34 33v6M41 8h6M44 5v6" }
  };

  var celestialId = 0;
  function detailedCelestial(name, svg) {
    var id = "planet-" + (++celestialId);
    svg.setAttribute("viewBox", "0 0 160 120");
    if (name === "gull" || name === "bird" || name === "feather") {
      svg.classList.add("star-cluster-art");
      svg.innerHTML = '<circle cx="44" cy="32" r="1.4"/><circle cx="106" cy="77" r="2"/><circle cx="126" cy="24" r="1"/><path d="M78 43L80 51L88 53L80 55L78 63L76 55L68 53L76 51Z"/>';
    } else if (name === "cloud" || name === "cloudline") {
      svg.classList.add("meteor-art");
      svg.setAttribute("viewBox", "0 0 200 80");
      svg.innerHTML = '<defs><linearGradient id="' + id + '-trail"><stop stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#e3e6ff"/></linearGradient></defs>' +
        '<path class="meteor-guide" d="M185 5L25 76"/>' +
        '<g class="meteor-flight"><path class="meteor-tail" style="stroke:url(#' + id + '-trail)" d="M65 -29L0 0"/><circle class="meteor-head" r="1.8"/></g>';
    } else return false;
    return true;
  }

  function icon(name) {
    var night = root.dataset.theme === "dark";
    var spec = (night && NIGHT_ICONS[name]) || ICONS[name];
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("data-sky-icon", name);
    if (night) svg.classList.add("celestial-icon");
    svg.setAttribute("viewBox", spec.viewBox);
    svg.setAttribute("aria-hidden", "true");
    if (night && detailedCelestial(name, svg)) return svg;
    (spec.parts || [["", spec.d]]).forEach(function (part) {
      var path = document.createElementNS(SVG_NS, "path");
      if (part[0]) path.setAttribute("class", part[0]);
      path.setAttribute("d", part[1]);
      svg.appendChild(path);
    });
    return svg;
  }

  // Gull, bird, feather and cloud placeholders get their SVG plus a wrapper that hovers gently.
  function hydrateIcons(scope) {
    var kinds = ["gull", "bird", "feather", "cloud", "cloudline"];
    var selector = kinds.map(function (k) { return ".fx-" + k + ":empty"; }).join(", ");
    scope.querySelectorAll(selector).forEach(function (el, i) {
      var inner = document.createElement("span");
      inner.style.setProperty("--hover", (5 + (i % 4) * 1.3).toFixed(1) + "s");
      var name = kinds.filter(function (k) { return el.classList.contains("fx-" + k); })[0];
      inner.appendChild(icon(name));
      el.appendChild(inner);
    });
  }

  // spec: [classes, x, y, depth, extra]; extra may hold w, h, s, tilt, flip, text.
  function fxField(specs) {
    var field = h("div", { class: "fx-field", "aria-hidden": "true" });
    specs.forEach(function (spec) {
      var extra = spec[4] || {};
      var el = h("span", { class: "fx " + spec[0], "data-depth": String(spec[3]) }, extra.text ? [extra.text] : []);
      el.style.setProperty("--x", spec[1]);
      el.style.setProperty("--y", spec[2]);
      if (extra.w) el.style.setProperty("--w", extra.w);
      if (extra.h) el.style.setProperty("--h", extra.h);
      if (extra.s) el.style.setProperty("--s", extra.s);
      if (extra.tilt) el.style.setProperty("--tilt", extra.tilt);
      if (extra.flip) el.style.setProperty("--flip", "-1");
      field.appendChild(el);
    });
    hydrateIcons(field);
    return field;
  }

  var SECTION_FX = [
    function (label) {
      return [
        ["fx-label fx-hide-sm", "46%", "1.4rem", 0.12, { text: label }],
        ["fx-frame fx-hide-sm", "72%", "-1.25rem", 0.3, { w: "8.5rem", h: "4.25rem" }],
        ["fx-gull", "82%", "-2.5rem", 0.55, { s: "84px", tilt: "6deg", flip: true }],
        ["fx-cloud fx-hide-sm", "58%", "-2.75rem", 0.2, { s: "130px" }],
      ];
    },
    function (label) {
      return [
        ["fx-label fx-hide-sm", "46%", "1.4rem", 0.12, { text: label }],
        ["fx-ring fx-hide-sm", "86%", "-2.5rem", 0.22, { w: "6.5rem" }],
        ["fx-gull fx-hide-sm", "68%", "-2.25rem", 0.45, { s: "64px", tilt: "10deg" }],
        ["fx-cloudline", "40%", "0.4rem", 0.3, { s: "90px" }],
      ];
    },
    function (label) {
      return [
        ["fx-label fx-hide-sm", "46%", "1.4rem", 0.12, { text: label }],
        ["fx-gull", "84%", "-0.75rem", 0.62, { s: "78px", tilt: "-8deg", flip: true }],
        ["fx-cloud", "64%", "-1rem", 0.22, { s: "110px" }],
      ];
    },
  ];

  var INTRO_FX = [
    ["fx-slab", "58%", "1.5rem", 0.14, { w: "40%", h: "85%" }],
    ["fx-frame fx-hide-sm", "70%", "2.5rem", 0.3, { w: "9rem", h: "5rem" }],
    ["fx-label fx-hide-sm", "54%", "6.5rem", 0.2, { text: "// ARCHIVE" }],
    ["fx-gull", "82%", "1.25rem", 0.55, { s: "88px", tilt: "6deg", flip: true }],
    ["fx-cloud fx-hide-sm", "60%", "62%", 0.2, { s: "120px" }],
  ];

  var LIST_FX = [
    ["fx-label fx-vertical fx-hide-sm", "101.5%", "3rem", 0.2, { text: "INDEX // ENTRIES" }],
  ];

  // Shared dot pattern used as the blueprint fill inside gull silhouettes.
  function injectPatterns() {
    if (document.getElementById("fx-dotpat")) return;
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("width", "0");
    svg.setAttribute("height", "0");
    svg.setAttribute("aria-hidden", "true");
    svg.style.position = "absolute";
    svg.innerHTML =
      '<defs><pattern id="fx-dotpat" width="7" height="7" patternUnits="userSpaceOnUse">' +
      '<circle cx="1.5" cy="1.5" r=".75" fill="#3a6fd8" fill-opacity=".4"/></pattern></defs>';
    document.body.appendChild(svg);
  }

  // Foreground layer for list pages: sits above the cards, overlaps their
  // edges, and uses negative depth so it scrolls slightly faster than the page.
  var LIST_TOP_FX = [
    ["fx-gull", "84%", "5%", -0.12, { s: "96px", tilt: "-6deg", flip: true }],
    ["fx-cloud fx-hide-sm", "-1%", "22%", -0.06, { s: "150px" }],
    ["fx-gull fx-hide-sm", "-1%", "38%", -0.16, { s: "74px", tilt: "8deg" }],
    ["fx-target fx-hide-sm", "90%", "47%", -0.08, { w: "4rem", h: "2.25rem" }],
    ["fx-gull", "92%", "64%", -0.14, { s: "82px", tilt: "-10deg", flip: true }],
    ["fx-cloud", "84%", "80%", -0.05, { s: "130px" }],
    ["fx-gull fx-hide-sm", "36%", "93%", -0.18, { s: "70px", tilt: "6deg" }],
  ];



  // Wrap the sidebar photo in a stage; scrolling wipes it away to reveal the OC.
  // Sidebar: photo slot, then name/school, then the OC panel. CSS drives the
  // swap from --swap (scroll progress over the first stretch of the page).
  var ocItems = [];
  var ocMarks = null;

  // Callouts breathe with the scroll: each one swells, shrinks and drops out
  // on its own phase, so the set keeps changing as the page moves.
  function updateOc() {
    if (!ocItems.length) return;
    var y = window.scrollY;
    var sp = clamp(y / Math.max(1, root.scrollHeight - window.innerHeight), 0, 1);
    if (ocMarks) {
      // Scan line sits at the same height as the dot level inside the body.
      var scan = ocMarks.querySelector(".hud-scan");
      scan.setAttribute("transform", "translate(0 " + (186 * (1 - sp)).toFixed(2) + ")");
      scan.querySelector("text").textContent = ("00" + Math.round(sp * 100)).slice(-3);
      if (!reduceMotion) {
        ocMarks.querySelector(".hud-tilt").setAttribute("d",
          "M0 " + (40 + sp * 90).toFixed(1) + "L135.6 " + (150 - sp * 110).toFixed(1));
        ocMarks.querySelector(".hud-ruler").style.strokeDashoffset = (y * 0.03).toFixed(2);
        [].forEach.call(ocMarks.querySelectorAll(".hud-grid"), function (g, i) {
          var gx = +g.getAttribute("data-x"), gy = +g.getAttribute("data-y");
          var cols = +g.getAttribute("data-cols"), rows = +g.getAttribute("data-rows");
          var dir = i % 2 ? -1 : 1;
          // grid lines slide; the lit cell steps through the patch
          ocMarks.querySelectorAll("pattern")[i].setAttribute("patternTransform",
            "translate(" + gx + " " + (gy + ((dir * y * 0.02) % 5)).toFixed(2) + ")");
          var n = Math.floor(y / 60 + i * 7) % (cols * rows);
          var cell = g.querySelector(".hud-cell");
          cell.setAttribute("x", gx + (n % cols) * 5);
          cell.setAttribute("y", gy + (dir > 0 ? Math.floor(n / cols) : rows - 1 - Math.floor(n / cols)) * 5);
        });
        [].forEach.call(ocMarks.querySelectorAll(".hud-hbar"), function (bar, i) {
          bar.setAttribute("width", (8 + 6.5 * (1 + Math.sin(y / 340 + i * 2.1))).toFixed(2));
        });
      }
    }
    if (reduceMotion) return;
    ocItems.forEach(function (item, i) {
      var v = Math.sin(y / 420 + i * 1.7 + 0.9);
      item.style.opacity = clamp((v + 0.8) / 0.4, 0, 1).toFixed(3);
      var zoom = item.querySelector(".hud-zoom");
      if (!zoom) return;
      var cx = item.getAttribute("data-cx"), cy = item.getAttribute("data-cy");
      zoom.setAttribute("transform", "translate(" + cx + " " + cy + ") scale(" + (1 + 0.16 * v).toFixed(3) +
        ") translate(-" + cx + " -" + cy + ")");
    });
  }

  function initAvatar() {
    var avatar = scope.querySelector(".sidebar .avatar");
    if (!avatar || avatar.parentNode.classList.contains("avatar-slot")) return;
    var top = avatar.parentNode;
    var slot = h("div", { class: "avatar-slot" });
    top.insertBefore(slot, avatar);
    slot.appendChild(avatar);
    // The OC art comes in layers: wings behind, the figure, three sparkles in
    // front. Wings and sparkles drift on their own (CSS). A blueprint-style
    // overlay frames the eyes and repeats them enlarged in a callout.
    var base = "/theme/";
    function layer(name, cls, alt) {
      return h("img", { src: base + name + ".webp", class: cls, alt: alt || "", decoding: "sync" });
    }
    // The bare body is a window (CSS mask): it fills with colour as the page
    // is read, and glows where the pointer is.
    // Seen through the body: the same light line-art as the page background.
    // Blueprint dots rise with reading progress; clouds, wind streaks and
    // gulls drift across; a faint wash follows the pointer.
    function ocWindow() {
      var drift = [
        ["fx-cloud", "16%", "46px", "19s", "-4s"],
        ["fx-cloudline", "34%", "40px", "15s", "-9s"],
        ["fx-gull", "50%", "30px", "23s", "-14s"],
        ["fx-cloud", "66%", "38px", "21s", "-1s"],
        ["fx-cloudline", "84%", "34px", "17s", "-11s"],
      ].map(function (d) {
        var el = h("span", { class: "oc-drift " + d[0] });
        el.style.setProperty("--y", d[1]);
        el.style.setProperty("--s", d[2]);
        el.style.setProperty("--dur", d[3]);
        el.style.animationDelay = d[4];
        return el;
      });
      var fill = h("div", { class: "oc-fill", "aria-hidden": "true" },
        [h("div", { class: "oc-level" })].concat(drift, [h("div", { class: "oc-glow" })]));
      hydrateIcons(fill);
      return fill;
    }
    var oc = h("div", { class: "oc" }, [
      h("div", { class: "oc-disc", "aria-hidden": "true" }),
      // colour layer of the wing: the same four feathers as flat blocks, on the
      // same roots and at the same angles, set a little further out
      layer("oc-fan1", "oc-wing oc-wing-1 oc-fan"),
      layer("oc-fan2", "oc-wing oc-wing-2 oc-fan"),
      layer("oc-fan3", "oc-wing oc-wing-3 oc-fan"),
      layer("oc-fan4", "oc-wing oc-wing-4 oc-fan"),
      // four feathers, outermost first; CSS fans them open with page progress
      layer("oc-wing1", "oc-wing oc-wing-1"),
      layer("oc-wing2", "oc-wing oc-wing-2"),
      layer("oc-wing3", "oc-wing oc-wing-3"),
      layer("oc-wing4", "oc-wing oc-wing-4"),
      layer("oc", "oc-body", "Scabbard, original character"),
      ocWindow(),
      layer("oc-star1", "oc-star oc-star-1"),
      layer("oc-star2", "oc-star oc-star-2"),
      layer("oc-star3", "oc-star oc-star-3"),
    ]);
    var hud = document.createElementNS(SVG_NS, "svg");
    hud.setAttribute("class", "oc-hud");
    hud.setAttribute("viewBox", "0 0 135.6 186");
    hud.setAttribute("aria-hidden", "true");
    // Handles: small squares on the corners and top/bottom midpoints of a box.
    function handles(x, y, w, hgt, size) {
      return [[x, y], [x + w, y], [x, y + hgt], [x + w, y + hgt], [x + w / 2, y], [x + w / 2, y + hgt]]
        .map(function (pt) {
          return '<rect class="hud-handle" x="' + (pt[0] - size / 2) + '" y="' + (pt[1] - size / 2) +
            '" width="' + size + '" height="' + size + '"/>';
        })
        .join("");
    }
    // A framed thumbnail from the colour illustration, with a label underneath
    // and a leader line to the matching spot on the figure. The frame is its own
    // group so scrolling can zoom it; the whole item fades as one.
    function callout(name, x, y, w, hgt, label, anchorEnd, from, to) {
      return '<g class="hud-item" data-cx="' + (x + w / 2) + '" data-cy="' + (y + hgt / 2) + '">' +
        (from ? '<path class="hud-leader" d="M' + from.join(" ") + "L" + to.join(" ") + '"/>' +
          '<circle class="hud-dot" cx="' + to[0] + '" cy="' + to[1] + '" r="1"/>' : "") +
        '<g class="hud-zoom"><image href="' + base + name + '.webp" x="' + x + '" y="' + y + '" width="' + w +
        '" height="' + hgt + '" preserveAspectRatio="none"/><rect class="hud-select" x="' + x + '" y="' + y +
        '" width="' + w + '" height="' + hgt + '"/>' + handles(x, y, w, hgt, 2) +
        '<text class="hud-label" x="' + (anchorEnd ? x + w : x) + '" y="' + (y + hgt + 5.2) + '"' +
        (anchorEnd ? ' text-anchor="end"' : "") + ">" + label + "</text></g></g>";
    }
    // Construction marks that move with the scroll (see updateOc): two grid
    // patches whose lines slide and whose lit cell steps along, a ruler, a
    // tilting guide line, and a scan line at the reading-progress height.
    function grid(id, x, y, cols, rows) {
      return '<pattern id="' + id + '" width="5" height="5" patternUnits="userSpaceOnUse">' +
        '<path class="hud-grid-line" d="M5 0H0V5"/></pattern>' +
        '<g class="hud-grid" data-x="' + x + '" data-y="' + y + '" data-cols="' + cols + '" data-rows="' + rows + '">' +
        '<rect class="hud-cell" x="' + x + '" y="' + y + '" width="5" height="5"/>' +
        '<rect class="hud-grid-box" x="' + x + '" y="' + y + '" width="' + cols * 5 + '" height="' + rows * 5 +
        '" fill="url(#' + id + ')"/></g>';
    }
    var marks =
      '<path class="hud-tilt" d="M0 60L135.6 130"/>' +
      grid("oc-grid-a", 4, 6, 6, 5) + grid("oc-grid-b", 103, 128, 6, 8) +
      '<path class="hud-ruler" d="M2 44V94"/><path class="hud-ruler-line" d="M0.6 44V94"/>' +
      '<g class="hud-scan"><path d="M34 0H102M34 -1.6V1.6M102 -1.6V1.6"/>' +
      '<text class="hud-label" x="35.5" y="-1.6">000</text></g>';
    // Solid pieces right of the arm: a tag with three data bars (updateOc
    // changes their lengths with the scroll) and an edge marker.
    var solids =
      '<rect class="hud-solid" x="112" y="54" width="21" height="6.4"/>' +
      '<text class="hud-tag-text" x="113.6" y="58.9">OC-01</text>' +
      ["hud-solid", "hud-solid-mid", "hud-solid-light"].map(function (c, i) {
        return '<rect class="hud-hbar ' + c + '" x="112" y="' + (63 + i * 4.6) + '" width="' + (20 - i * 5) + '" height="3"/>';
      }).join("") +
      '<path class="hud-solid-mid" d="M135.6 84l-5.5 3.6 5.5 3.6Z"/>';
    hud.innerHTML = marks + solids +
      // dotted range arc with tick labels, lower left
      '<g class="hud-item"><path class="hud-dots" d="M14 184A160 160 0 0 1 10 144"/>' +
      '<g class="hud-text"><circle cx="9.6" cy="152" r="0.9"/><text x="12.6" y="153.7">A1</text>' +
      '<circle cx="10.4" cy="163" r="0.9"/><text x="13.4" y="164.7">A2</text>' +
      '<circle cx="11.9" cy="174" r="0.9"/><text x="14.9" y="175.7">A3</text></g></g>' +
      callout("oc-eyes", 78, -17, 50, 16.1, "OC-01 // EYE", true, [78, -7], [73.4, 23.2]) +
      // the top sits right on the chest, no leader needed
      callout("oc-top", 60.3, 47.5, 21, 21, "OC-02 // TOP", false) +
      callout("oc-shorts", 106, 98, 26, 19.5, "OC-03 // SHORTS", true, [106, 106], [80.4, 85]) +
      callout("oc-cape", 2, 112, 30, 25, "OC-04 // CAPE", false, [14, 112], [20, 72]);
    oc.appendChild(hud);
    ocItems = [].slice.call(hud.querySelectorAll(".hud-item"));
    ocMarks = hud;
    // Pointer anywhere over the sidebar lights the body where it is closest.
    var side = top.closest ? top.closest(".sidebar") : null;
    // Construction marks around the name block and beside the links. They are
    // plain spans; CSS moves them from --sp (page progress).
    function sideMarks(cls, names) {
      return h("div", { class: "side-marks " + cls, "aria-hidden": "true" },
        names.map(function (n) { return h("span", { class: n }, n === "sm-coord" ? ["32.88\u00b0N 117.23\u00b0W"] : []); }));
    }
    var info = slot.nextElementSibling;
    if (info) {
      info.classList.add("side-info");
      info.appendChild(sideMarks("side-marks-top", ["sm-corner sm-tl", "sm-corner sm-br", "sm-rule", "sm-cross", "sm-coord"]));
    }
    var links = side && side.querySelector(".social-nav");
    if (links) links.appendChild(sideMarks("side-marks-bottom", ["sm-vline", "sm-grid", "sm-ring", "sm-seg"]));
    if (side) {
      side.addEventListener("pointermove", function (ev) {
        var r = oc.getBoundingClientRect();
        if (!r.width || !r.height) return;
        oc.style.setProperty("--mx", (((ev.clientX - r.left) / r.width) * 100).toFixed(1) + "%");
        oc.style.setProperty("--my", (((ev.clientY - r.top) / r.height) * 100).toFixed(1) + "%");
        oc.classList.add("is-near");
      });
      side.addEventListener("pointerleave", function () {
        oc.classList.remove("is-near");
      });
    }
    top.appendChild(h("div", { class: "oc-slot" }, [h("div", { class: "oc-reveal" }, [oc])]));
  }

  function initDecor() {
    initAvatar();
    injectPatterns();
    var sidebar = scope.querySelector(".sidebar");
    if (sidebar && !sidebar.querySelector(".avatar-birds")) {
      var avatarBirds = h("span", { class: "sidebar-birds avatar-birds", "aria-hidden": "true" });
      avatarBirds.appendChild(icon("gull"));
      avatarBirds.appendChild(icon("bird"));
      sidebar.appendChild(avatarBirds);
    }
    scope.querySelectorAll(".side-info, .social-nav").forEach(function (panel) {
      if (panel.querySelector(".sidebar-birds")) return;
      var birds = h("span", { class: "sidebar-birds", "aria-hidden": "true" });
      birds.appendChild(icon("gull"));
      birds.appendChild(icon("bird"));
      panel.appendChild(birds);
    });
    hydrateIcons(scope);
    scope.querySelectorAll(".home-section").forEach(function (section, index) {
      var heading = section.querySelector("h3");
      var name = heading ? heading.textContent.trim().toUpperCase() : "";
      var label = "0" + (index + 1) + " // " + name;
      section.insertBefore(fxField(SECTION_FX[index % SECTION_FX.length](label)), section.firstChild);
    });

    scope.querySelectorAll(".page-intro:not(.hero-copy)").forEach(function (intro) {
      intro.insertBefore(fxField(INTRO_FX), intro.firstChild);
    });

    scope.querySelectorAll(".post-wrap").forEach(function (wrap) {
      wrap.insertBefore(fxField(LIST_FX), wrap.firstChild);
      var top = fxField(LIST_TOP_FX);
      top.classList.add("fx-over");
      wrap.appendChild(top);
    });

    scope.querySelectorAll(".fx-field").forEach(function (field, fieldIndex) {
      if (field.querySelector(".night-dust")) return;
      var dust = h("div", { class: "night-dust", "aria-hidden": "true" });
      for (var i = 0; i < 56; i++) {
        var star = h("i", {});
        star.style.cssText = "left:" + ((i * 61.803 + fieldIndex * 11) % 98) + "%;top:" + ((i * 43.719 + 9) % 97) + "%;--twinkle:" + (3 + i % 5) + "s;--phase:-" + (i * 0.73) + "s";
        dust.appendChild(star);
      }
      field.appendChild(dust);
    });

    scope.querySelectorAll(".hero > .fx-field, .page-intro > .fx-field, .post-wrap > .fx-field:not(.fx-over)").forEach(function (field, index) {
      if (field.querySelector(".fx-satellite")) return;
      var satellite = h("span", { class: "fx fx-satellite", "data-depth": "0.12" });
      satellite.style.cssText = "--x:" + (index ? "88%" : "80%") + ";--y:" + (index ? "20rem" : "62%") + ";--sat-tilt:" + (index ? "18deg" : "-18deg");
      var craft = h("span", { class: "satellite-drift" });
      craft.appendChild(icon("satellite"));
      satellite.appendChild(craft);
      field.appendChild(satellite);
    });

    var sky = scope.querySelector(".hero > .fx-field, .page-intro > .fx-field");
    if (sky && !sky.querySelector(".polar-trails")) {
      var polar = document.createElementNS(SVG_NS, "svg");
      polar.setAttribute("class", "polar-trails");
      polar.setAttribute("viewBox", "0 0 600 600");
      polar.setAttribute("aria-hidden", "true");
      var orbit = document.createElementNS(SVG_NS, "g");
      orbit.setAttribute("class", "polar-orbits");
      // A single angular velocity preserves the shared celestial pole.
      for (var ring = 0; ring < 16; ring++) {
        var radius = 35 + ring * 16;
        var phase = ring * 47;
        var group = document.createElementNS(SVG_NS, "g");
        group.setAttribute("transform", "rotate(" + phase + " 300 300)");
        var trail = document.createElementNS(SVG_NS, "circle");
        trail.setAttribute("cx", "300"); trail.setAttribute("cy", "300");
        trail.setAttribute("r", String(radius));
        trail.setAttribute("pathLength", "360");
        trail.setAttribute("stroke-dasharray", (38 + ring % 5 * 13) + " " + (100 + ring % 3 * 17) + " 18 360");
        trail.style.opacity = String(0.25 + ring % 4 * 0.13);
        group.appendChild(trail);
        var star = document.createElementNS(SVG_NS, "circle");
        star.setAttribute("class", "polar-star");
        star.setAttribute("cx", String(300 + radius)); star.setAttribute("cy", "300");
        star.setAttribute("r", ring % 4 === 0 ? "2" : "1.2");
        star.style.setProperty("--phase", (-ring * 0.7) + "s");
        group.appendChild(star);
        orbit.appendChild(group);
      }
      polar.appendChild(orbit);
      var pole = document.createElementNS(SVG_NS, "path");
      pole.setAttribute("class", "polar-origin");
      pole.setAttribute("d", "M300 293L301.6 298.4L307 300L301.6 301.6L300 307L298.4 301.6L293 300L298.4 298.4Z");
      polar.appendChild(pole);
      sky.appendChild(polar);
    }

    var nav = scope.querySelector(".top-nav");
    if (nav && !nav.querySelector(".theme-toggle")) {
      var toggle = h("button", { class: "theme-toggle", type: "button" });
      function updateToggle() {
        var dark = root.dataset.theme === "dark";
        toggle.textContent = dark ? "\u2600" : "\u263e";
        toggle.setAttribute("aria-label", dark ? "Switch to day mode" : "Switch to night mode");
        toggle.setAttribute("aria-pressed", String(dark));
        toggle.title = dark ? "Day mode" : "Night mode";
      }
      function applyTheme() {
        root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
        try { localStorage.setItem("site-theme", root.dataset.theme); } catch (error) {}
        scope.querySelectorAll("svg[data-sky-icon]").forEach(function (svg) {
          svg.replaceWith(icon(svg.getAttribute("data-sky-icon")));
        });
        updateToggle();
      }
      var themeTransition = null;
      toggle.addEventListener("click", function () {
        if (themeTransition) return;
        if (reduceMotion || !document.startViewTransition) {
          applyTheme();
          return;
        }
        root.classList.add("theme-curtain");
        themeTransition = document.startViewTransition(applyTheme);
        themeTransition.finished.catch(function () {}).finally(function () {
          root.classList.remove("theme-curtain");
          themeTransition = null;
        });
      });
      updateToggle();
      nav.appendChild(toggle);
    }
    if (nav && !nav.querySelector(".nav-flight")) {
      var bird = h("span", { class: "nav-bird" });
      bird.appendChild(icon("bird"));
      nav.appendChild(
        h("div", { class: "nav-flight", "aria-hidden": "true" }, [h("span", { class: "nav-flight-line" }), bird])
      );
    }
  }

  // Small seeded generator so the scattered layout is the same on every visit.
  function seeded(seed) {
    return function () {
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Overlay pieces: a few distinct shapes only, so lists don't get busy.
  var OVERLAY_KINDS = [
    function (r) { return ["fx-gull", { s: Math.round(44 + r() * 24) + "px", tilt: Math.round(r() * 24 - 12) + "deg", flip: r() < 0.5 }]; },
    function (r) { return ["fx-cloud", { s: Math.round(64 + r() * 40) + "px" }]; },
    function (r) { return ["fx-cloudline", { s: Math.round(60 + r() * 30) + "px" }]; },
    function (r) { return ["fx-dash", { w: Math.round(80 + r() * 120) + "px" }]; },
  ];
  // Scatter small pieces over a rendered list; they sit above the items and
  // drift at their own depth while scrolling.
  // gullEvery: when set, every Nth piece is a larger gull (used on list pages).
  function sprinkle(list, name, seed, gullEvery) {
    if (!list || list.querySelector(":scope > .fx-over")) return;
    var r = seeded(seed);
    var count = Math.max(2, Math.round(list.offsetHeight / 320));
    var specs = [];
    for (var i = 0; i < count; i += 1) {
      var kind = OVERLAY_KINDS[(i * 4 + seed) % OVERLAY_KINDS.length];
      var label = "// " + name + " " + String(i + 1).padStart(2, "0");
      var made = kind(r, label);
      if (gullEvery && i % gullEvery === 0) {
        made = ["fx-gull", { s: Math.round(60 + r() * 30) + "px", tilt: Math.round(r() * 20 - 10) + "deg", flip: r() < 0.5 }];
      }
      var y = ((i + 0.25 + r() * 0.5) / count) * 100;
      var x = 16 + r() * 78;
      var depth = r() < 0.25 ? -(0.08 + r() * 0.12) : 0.15 + r() * 0.5;
      var extra = made[1] || {};
      specs.push([made[0] + (i % 2 ? " fx-hide-sm" : ""), x.toFixed(1) + "%", y.toFixed(1) + "%", depth.toFixed(2), extra]);
    }
    var field = fxField(specs);
    field.classList.add("fx-over");
    // Continue the night-sky composition down the rendered archive.
    var extraCount = list.offsetHeight > 1600 ? 2 : 1;
    for (var j = 0; j < extraCount; j++) {
      var craft = h("span", { class: "fx fx-satellite satellite-extra", "data-depth": "-0.04" });
      var satelliteX = j % 2 ? 4 + r() * 26 : 71 + r() * 27;
      var satelliteY = (j + 0.2 + r() * 0.6) / extraCount * 100;
      craft.style.cssText = "--x:" + satelliteX.toFixed(1) + "%;--y:" + satelliteY.toFixed(1) +
        "%;--sat-tilt:" + Math.round(r() * 300 - 150) + "deg;--sat-size:" + (4.5 + r() * 8).toFixed(1) +
        "rem;--sat-duration:" + (13 + r() * 12).toFixed(1) + "s;--sat-delay:-" + (r() * 20).toFixed(1) + "s";
      var drift = h("span", { class: "satellite-drift" });
      drift.appendChild(icon("satellite"));
      craft.appendChild(drift);
      field.appendChild(craft);
      var original = scope.querySelector(".polar-trails");
      if (original && j < 2) {
        var trails = original.cloneNode(true);
        trails.classList.add("polar-extra");
        trails.style.cssText = "left:" + (j % 2 ? "85%" : "12%") + ";top:" + (extraCount === 1 ? 45 : 24 + j * 48) + "%";
        field.appendChild(trails);
      }
    }
    if (list.matches("[data-item-list], [data-home-catalog*='publication'] .home-list")) {
      var leftCraft = h("span", { class: "fx fx-satellite satellite-extra", "data-depth": "0.06" });
      leftCraft.style.cssText = "--x:9%;--y:72%;--sat-tilt:32deg;--sat-size:6.5rem;--sat-duration:21s;--sat-delay:-9s";
      var leftDrift = h("span", { class: "satellite-drift" });
      leftDrift.appendChild(icon("satellite"));
      leftCraft.appendChild(leftDrift);
      field.appendChild(leftCraft);
    }
    list.appendChild(field);
    registerLayers(field);
  }

  /* ---------------- Scroll: parallax + progress ---------------- */

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  var layers = [];

  function registerLayers(scope) {
    if (reduceMotion) return;
    scope.querySelectorAll("[data-depth]").forEach(function (el) {
      if (el._layer) return;
      el._layer = true;
      layers.push(el);
    });
    requestScrollUpdate();
  }

  function initScroll() {
    var nav = scope.querySelector(".top-nav");
    var ticking = false;

    function measureNav() {
      root.style.setProperty("--nav-h", (nav ? nav.offsetHeight : 0) + "px");
    }

    // Each layer drifts by its depth as it travels: positive depth trails the
    // scroll (far away), negative depth leads it (close up). A layer sits at its
    // designed spot when it crosses the viewport centre, or at page load if it
    // is already above the centre then, so nothing near the top starts displaced.
    function updateParallax() {
      var mid = window.innerHeight / 2;
      var scrollY = window.scrollY;
      var moves = layers.map(function (el) {
        var rect = el.getBoundingClientRect();
        var centre = rect.top + rect.height / 2 - (el._py || 0);
        var rest = Math.min(centre + scrollY, mid);
        var depth = parseFloat(el.getAttribute("data-depth")) || 0;
        return clamp(-(centre - rest) * depth, -200, 200);
      });
      layers.forEach(function (el, i) {
        el._py = moves[i];
        el.style.translate = "0 " + moves[i].toFixed(1) + "px";
      });
    }

    function update() {
      ticking = false;
      var max = Math.max(1, root.scrollHeight - window.innerHeight);
      root.style.setProperty("--sp", clamp(window.scrollY / max, 0, 1).toFixed(4));
      // Photo-to-OC wipe completes over the first stretch of scrolling.
      root.style.setProperty("--swap", clamp(window.scrollY / Math.min(320, max * 0.6), 0, 1).toFixed(4));
      updateOc();
      checkReveals();
      if (layers.length) updateParallax();
    }

    function request() {
      if (ticking) return;
      ticking = true;
      scrollFrame = window.requestAnimationFrame(function () { if (!disposed) update(); });
    }

    requestScrollUpdate = request;
    registerLayers(scope);
    measureNav();
    update();
    window.addEventListener("scroll", request, { passive: true, signal: controller.signal });
    window.addEventListener("resize", function () {
      measureNav();
      request();
    }, { signal: controller.signal });
    const resize = new ResizeObserver(request);
    resize.observe(scope);
    controller.signal.addEventListener("abort", function () { resize.disconnect(); });
  }

  function h(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      var value = attrs[key];
      if (value == null || value === false) return;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    });
    (children || []).forEach(function (child) {
      if (child == null || child === false) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }


  initDecor();
  scope.querySelectorAll(".post-list").forEach(function (list, index) {
    sprinkle(list, "NOTES", 1500 + index, 3);
  });
  initScroll();
  const inserted = Array.from(scope.querySelectorAll("*")).filter(el => !before.has(el) && before.has(el.parentElement));
  return function () {
    disposed = true;
    controller.abort();
    cancelAnimationFrame(scrollFrame);
    inserted.forEach(el => { if (!side?.contains(el)) el.remove(); });
    if (side && sideMarkup !== undefined) side.innerHTML = sideMarkup;
  };
}
