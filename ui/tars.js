"use strict";
(() => {
  const RM = Live.rm, $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const C = {fondo: "#18225a", hondo: "#121a47", lino: "#f1ead6", lino2: "#c3c1d4", x: "#a9bdff", z: "#ff8da0", sint: "#f6bd44", ok: "#5fd3a8", mal: "#ff5a74"};
  const SER = {BB72: C.lino, BB90: C.x, BB108: C.sint, BB144: C.z, BB288: C.ok};
  const nf = Live.nf, num = (el, ns) => { const t = Live.t(ns); el.innerHTML = `${t.v}<small>${t.u}</small>`; };
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; e.innerHTML = html || ""; return e; };
  const loc = n => Number(n).toLocaleString("es-AR");
  const dTxt = c => c.exact ? `d=${c.d_x}` : `d entre ${c.d_x_lower} y ${c.d_x}`;   // BB288: solo cotas, sin certificar

  /* =====================================================  EL TEJIDO  ===================================================== */
  /* Un lienzo por código: clavos = qubits, cuadrados = chequeos X, rombos = chequeos Z. Los hilos que se tensan son los síntomas. */
  const ease = t => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
  class Weave {
    constructor(cv, {compact, tip}) {
      this.cv = cv; this.g = cv.getContext("2d"); this.box = cv.parentElement; this.compact = compact; this.tip = tip || null;
      this.code = null; this.geo = null; this.base = null; this.hov = null; this.layer = null; this.dirty = true; this.playing = false; this.W = 0; this.H = 0; this.dpr = 1;
      this.live = {t0: -1e9, zs: [], xs: [], errX: [], errZ: [], corrX: [], corrZ: []};
      new ResizeObserver(() => this.layout()).observe(this.box);
      if (!compact) { cv.addEventListener("pointermove", e => this.point(e)); cv.addEventListener("pointerleave", () => { this.hov = null; if (this.tip) this.tip.style.opacity = 0; this.dirty = true; }); }
    }
    setCode(c) { this.code = c; this.clear(); this.layout(); }
    clear() { Object.assign(this.live, {t0: -1e9, zs: [], xs: [], errX: [], errZ: [], corrX: [], corrZ: []}); this.dirty = true; }
    show(d) { Object.assign(this.live, {t0: performance.now(), zs: d.z_syndrome || [], xs: d.x_syndrome || [], errX: d.err_x || [], errZ: d.err_z || [], corrX: d.corr_x || [], corrZ: d.corr_z || []}); this.dirty = true; }
    layout() {
      const r = this.box.getBoundingClientRect(); if (r.width < 4 || r.height < 4) return;
      this.dpr = Math.min(devicePixelRatio || 1, 2); this.W = r.width; this.H = r.height;
      this.cv.width = this.W * this.dpr; this.cv.height = this.H * this.dpr; this.g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); this.build(); this.dirty = true;
    }
    build() {
      const code = this.code; if (!code) return; const T = code.tanner, n = code.n, mx = T.x_checks.length, mz = T.z_checks.length, {W, H} = this;
      const R = Math.min(W, H) * (this.compact ? .44 : .40), cx = this.compact ? W / 2 : (W > H * 1.25 ? W * .64 : W / 2), cy = H / 2;
      const pt = (r, a) => [cx + Math.cos(a) * r, cy + Math.sin(a) * r], ang = (i, m, off) => -Math.PI / 2 + (i + off) / m * Math.PI * 2;
      const geo = {R, cx, cy, D: Array.from({length: n}, (_, q) => pt(R, ang(q, n, 0))), X: Array.from({length: mx}, (_, c) => pt(R * .42, ang(c, mx, .5))), Z: Array.from({length: mz}, (_, c) => pt(R * .66, ang(c, mz, .25))), xe: T.x_edges, ze: T.z_edges};
      geo.nx = geo.X.map(() => []); geo.nz = geo.Z.map(() => []); geo.qx = geo.D.map(() => []); geo.qz = geo.D.map(() => []);
      geo.xe.forEach(([c, q]) => { geo.nx[c].push(q); geo.qx[q].push(c); }); geo.ze.forEach(([c, q]) => { geo.nz[c].push(q); geo.qz[q].push(c); });
      this.geo = geo;
      const base = document.createElement("canvas"); base.width = W * this.dpr; base.height = H * this.dpr; const b = base.getContext("2d"); b.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      b.lineWidth = this.compact ? Math.max(.5, Math.min(.9, 220 / n * .5)) : Math.max(.7, Math.min(1.2, 300 / n * .5));
      b.strokeStyle = "rgba(169,189,255,.26)"; b.beginPath(); geo.xe.forEach(([c, q]) => { b.moveTo(...geo.X[c]); b.lineTo(...geo.D[q]); }); b.stroke();
      b.strokeStyle = "rgba(255,141,160,.26)"; b.beginPath(); geo.ze.forEach(([c, q]) => { b.moveTo(...geo.Z[c]); b.lineTo(...geo.D[q]); }); b.stroke();
      b.strokeStyle = "rgba(241,234,214,.30)"; b.lineWidth = 1; b.beginPath(); b.arc(cx, cy, R + (this.compact ? 7 : 14), 0, 7); b.stroke();
      const nr = this.compact ? Math.max(1, Math.min(2.2, 260 / n)) : Math.max(1.6, Math.min(3.6, 560 / n)); geo.nr = nr;
      b.fillStyle = C.lino; geo.D.forEach(([x, y]) => { b.beginPath(); b.arc(x, y, nr, 0, 7); b.fill(); });
      const s = this.compact ? 2.4 : 3.6, d = this.compact ? 3.4 : 5;
      geo.X.forEach(([x, y]) => { b.fillStyle = C.fondo; b.strokeStyle = C.x; b.lineWidth = 1.5; b.fillRect(x - s, y - s, 2 * s, 2 * s); b.strokeRect(x - s, y - s, 2 * s, 2 * s); });
      geo.Z.forEach(([x, y]) => { b.fillStyle = C.fondo; b.strokeStyle = C.z; b.lineWidth = 1.5; b.beginPath(); b.moveTo(x, y - d); b.lineTo(x + d, y); b.lineTo(x, y + d); b.lineTo(x - d, y); b.closePath(); b.fill(); b.stroke(); });
      this.base = base;
    }
    edges(A, ed, ids, col, w, a) { const g = this.g; g.strokeStyle = col; g.lineWidth = w; g.globalAlpha = a; g.beginPath(); ed.forEach(([c, q]) => { if (ids.has(c)) { g.moveTo(...A[c]); g.lineTo(...this.geo.D[q]); } }); g.stroke(); g.globalAlpha = 1; }
    ring([x, y], r, col) { const g = this.g; g.strokeStyle = col; g.beginPath(); g.arc(x, y, r, 0, 7); g.stroke(); }
    frame(now) {
      const L = this.live, age = (now - L.t0) / 1000, fresh = age < 5.5;
      if (!this.dirty && !fresh && !this.playing && !this.hov) return; this.dirty = false;
      const g = this.g, geo = this.geo, k = this.compact ? .6 : 1; g.clearRect(0, 0, this.W, this.H); if (!geo || !this.base) return;
      g.drawImage(this.base, 0, 0, this.W, this.H);
      if (this.layer) {
        const ex = [], ez = []; this.layer.forEach(([t, c, q]) => (t === "X" ? ex : ez).push([c, q]));
        g.lineWidth = 2; g.strokeStyle = C.x; g.beginPath(); ex.forEach(([c, q]) => { g.moveTo(...geo.X[c]); g.lineTo(...geo.D[q]); }); g.stroke();
        g.strokeStyle = C.z; g.beginPath(); ez.forEach(([c, q]) => { g.moveTo(...geo.Z[c]); g.lineTo(...geo.D[q]); }); g.stroke();
        g.fillStyle = C.lino; const ph = (now / 900) % 1, dot = (a, b) => { g.beginPath(); g.arc(a[0] + (b[0] - a[0]) * ph, a[1] + (b[1] - a[1]) * ph, 2.6, 0, 7); g.fill(); };
        ex.forEach(([c, q]) => dot(geo.X[c], geo.D[q])); ez.forEach(([c, q]) => dot(geo.D[q], geo.Z[c]));
      }
      const a1 = ease(age / .25) * (1 - ease((age - 3.2) / 2));
      if (a1 > .01) {
        const zs = new Set(L.zs), xs = new Set(L.xs);
        this.edges(geo.Z, geo.ze, zs, C.sint, 2.4 * k, a1); this.edges(geo.X, geo.xe, xs, C.sint, 2.4 * k, a1);
        g.fillStyle = C.sint; g.globalAlpha = a1;
        const pul = c => { const [x, y] = c; g.beginPath(); g.arc(x, y, (6 + 3 * Math.sin(age * 9) * (1 - ease(age / 1.2))) * k, 0, 7); g.fill(); };
        L.zs.forEach(c => pul(geo.Z[c])); L.xs.forEach(c => pul(geo.X[c])); g.globalAlpha = 1;
      }
      const a2 = 1 - ease((age - 3.6) / 1.6);
      if (a2 > .01) {
        g.lineWidth = 2; g.globalAlpha = a2;
        L.errX.forEach(q => this.ring(geo.D[q], (9 + 2 * Math.sin(age * 7)) * k, C.mal)); L.errZ.forEach(q => this.ring(geo.D[q], (9 + 2 * Math.sin(age * 7)) * k, C.x));
        const kk = ease((age - 1.1) / .5); if (kk > 0) { g.globalAlpha = a2 * kk;[...L.corrX, ...L.corrZ].forEach(q => this.ring(geo.D[q], 14 * k, C.ok)); }
        g.globalAlpha = 1;
      }
      if (!this.compact && a2 > .01) {
        g.font = "500 12px Martian"; g.textBaseline = "middle"; g.globalAlpha = a2;
        const lab = (q, txt, col, dist) => { const [x, y] = geo.D[q], dx = x - geo.cx, dy = y - geo.cy, m = Math.hypot(dx, dy) || 1, tx = x + dx / m * dist, ty = y + dy / m * dist;
          g.strokeStyle = col; g.lineWidth = 1; g.beginPath(); g.moveTo(x + dx / m * 12, y + dy / m * 12); g.lineTo(tx, ty); g.stroke();
          if (tx < this.W * .5 && ty > this.H * .68) return;   // ese rincón es del texto de lectura
          g.fillStyle = col; g.textAlign = dx < 0 ? "right" : "left"; g.fillText(txt, tx + (dx < 0 ? -6 : 6), ty); };
        [...L.errX.map(q => [q, C.mal]), ...L.errZ.map(q => [q, C.x])].slice(0, 4).forEach(([q, col]) => lab(q, this.W < 520 ? `q${q}` : `falla en q${q}`, col, this.W < 520 ? 22 : 34));
        if (ease((age - 1.1) / .5) > .5) [...L.corrX, ...L.corrZ].slice(0, 4).forEach(q => lab(q, this.W < 520 ? `q${q} ok` : `q${q} corregido`, C.ok, this.W < 520 ? 40 : 58));
        g.globalAlpha = 1; g.textBaseline = "alphabetic";
      }
      if (this.hov) {
        const h = this.hov; g.lineWidth = 2.2; g.strokeStyle = C.lino; g.beginPath();
        if (h.t === "q") { geo.qx[h.i].forEach(c => { g.moveTo(...geo.X[c]); g.lineTo(...geo.D[h.i]); }); geo.qz[h.i].forEach(c => { g.moveTo(...geo.Z[c]); g.lineTo(...geo.D[h.i]); }); }
        else { const P = h.t === "x" ? geo.X[h.i] : geo.Z[h.i], N = h.t === "x" ? geo.nx[h.i] : geo.nz[h.i]; N.forEach(q => { g.moveTo(...P); g.lineTo(...geo.D[q]); }); }
        g.stroke();
      }
    }
    point(e) {
      const geo = this.geo; if (!geo) return; const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; let best = null, bd = 14 * 14;
      const test = (arr, t) => arr.forEach(([px, py], i) => { const d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = {t, i}; } });
      test(geo.D, "q"); test(geo.X, "x"); test(geo.Z, "z");
      if ((best && this.hov && best.t === this.hov.t && best.i === this.hov.i) || (!best && !this.hov)) return; this.hov = best; this.dirty = true;
      if (!this.tip) return; if (!best) { this.tip.style.opacity = 0; return; }
      const P = best.t === "q" ? geo.D[best.i] : best.t === "x" ? geo.X[best.i] : geo.Z[best.i];
      this.tip.textContent = best.t === "q" ? `qubit ${best.i}: está en ${geo.qx[best.i].length} chequeos X y ${geo.qz[best.i].length} Z` : `chequeo ${best.t.toUpperCase()} ${best.i}: toca ${(best.t === "x" ? geo.nx : geo.nz)[best.i].length} qubits`;
      this.tip.style.left = P[0] + "px"; this.tip.style.top = P[1] + "px"; this.tip.style.opacity = 1;
    }
  }
  const weaves = new Set();
  (function loop(now) { requestAnimationFrame(loop); weaves.forEach(w => { if (w.box.offsetParent) w.frame(now); }); })(0);

  /* =====================================================  CINTA DE LATENCIAS  ===================================================== */
    const TICKS = [[100, "100 ns"], [1e3, "1 µs"], [1e4, "10 µs"], [1e5, "100 µs"], [1e6, "1 ms"], [1e7, "10 ms"]];
  class Strip {
    constructor(cv) { this.cv = cv; this.g = cv.getContext("2d"); this.series = []; new ResizeObserver(() => this.size()).observe(cv.parentElement); }
    size() { const r = this.cv.parentElement.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2); this.w = r.width - (this.cv.dataset.inset ? 0 : 0); this.h = r.height; this.cv.width = this.w * d; this.cv.height = this.h * d; this.g.setTransform(d, 0, 0, d, 0, 0); this.draw(); }
    draw() {
      const g = this.g, {w, h} = this; if (!w) return; g.clearRect(0, 0, w, h);
      let mx = 0; this.series.forEach(s => s.data.slice(-400).forEach(v => { if (v > mx) mx = v; }));
      const lo = Math.log10(100), hi = Math.log10(Math.min(1e7, Math.max(1e4, Math.pow(10, Math.ceil(Math.log10(Math.max(mx, 1)))) * 1.01)));
      const Lm = 58, B = 6, T = 22, pw = w - Lm - 8, ph = h - T - B, Y = ns => T + (1 - (Math.log10(Math.max(100, Math.min(Math.pow(10, hi), ns))) - lo) / (hi - lo)) * ph;
      g.font = "400 11px Martian"; g.textAlign = "right"; g.lineWidth = 1;
      TICKS.filter(([v]) => Math.log10(v) <= hi + 1e-9).forEach(([v, t]) => { const y = Y(v), us = v === 1e3; g.strokeStyle = us ? C.mal : "rgba(241,234,214,.14)"; g.lineWidth = us ? 1.6 : 1; g.beginPath(); g.moveTo(Lm, y); g.lineTo(Lm + pw, y); g.stroke(); g.fillStyle = us ? C.mal : C.lino2; g.fillText(t, Lm - 8, y + 3.5); });
      const step = Math.max(5, Math.min(14, pw / 90)), N = Math.floor(pw / step);
      this.series.forEach(s => { const d = s.data.slice(-N); if (!d.length) return; g.strokeStyle = s.color; g.lineWidth = s.w || 1.6; g.beginPath(); d.forEach((v, i) => { const x = Lm + pw - (d.length - 1 - i) * step, y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
        g.fillStyle = s.color; const x = Lm + pw, y = Y(d[d.length - 1]); g.beginPath(); g.arc(x, y, 3, 0, 7); g.fill(); });
      g.textAlign = "left"; g.font = "500 12px Familjen"; let x = Lm + 4; this.series.forEach(s => { g.fillStyle = s.color; g.fillRect(x, 8, 12, 3); g.fillStyle = C.lino; g.fillText(s.name, x + 18, 14); x += 28 + g.measureText(s.name).width; });
    }
  }

  /* =====================================================  ESTADO COMPARTIDO  ===================================================== */
  let codes = [], by = {}, view = "cinco", bigName = null;
  const shown = new Set(); let pf = .5; const P0 = -3, P1 = -1; const curP = () => Math.pow(10, P0 + pf * (P1 - P0));
  const rails = [];
  function makeRail(el, label) {
    el.innerHTML = '<span class="pe"></span>';
    for (let e = -3; e <= -1; e++) for (let k = 1; k < 10; k++) { const f = (Math.log10(k * Math.pow(10, e)) - P0) / (P1 - P0); if (f > 1) continue; const t = document.createElement("i"); t.className = "tk"; t.style.left = f * 100 + "%"; if (k > 1) { t.style.height = "6px"; t.style.top = "12px"; } el.appendChild(t); }
    const pe = $(".pe", el); el.setAttribute("role", "slider"); el.setAttribute("tabindex", "0"); el.setAttribute("aria-label", label); el.setAttribute("aria-valuemin", "0.001"); el.setAttribute("aria-valuemax", "0.1");
    const ev = e => { const r = el.getBoundingClientRect(); setP((e.clientX - r.left) / r.width, true); };
    el.addEventListener("pointerdown", e => { el.setPointerCapture(e.pointerId); ev(e); const mv = x => ev(x), up = () => { el.removeEventListener("pointermove", mv); el.removeEventListener("pointerup", up); }; el.addEventListener("pointermove", mv); el.addEventListener("pointerup", up); });
    el.addEventListener("keydown", e => { if (e.key === "ArrowLeft") setP(pf - .04, true); if (e.key === "ArrowRight") setP(pf + .04, true); });
    rails.push({el, pe}); return el;
  }
  function setP(f, send) {
    pf = Math.min(1, Math.max(0, f)); const p = curP();
    rails.forEach(r => { r.pe.style.left = pf * 100 + "%"; r.el.setAttribute("aria-valuenow", p.toFixed(4)); }); $$(".pval").forEach(e => e.textContent = Live.sci(p));
    if (send) { clearTimeout(setP.t); setP.t = setTimeout(() => { if (view === "cinco") Object.values(five).forEach(c => c.ch.configure({p})); else if (big) big.ch.configure({p}); }, 150); }
  }

  /* =====================================================  VISTA DE LOS CINCO  ===================================================== */
  const five = {};   // nombre -> {ch, weave, hist, ...}
  const cardsEl = $("#cards"), chipsEl = $("#chips-codigos");
  const strip5 = new Strip($("#strip5"));
  function buildCards() {
    cardsEl.innerHTML = ""; chipsEl.innerHTML = "";
    codes.forEach(c => {
      const card = document.createElement("div"); card.className = "card"; card.dataset.c = c.name; card.tabIndex = 0; card.setAttribute("role", "button"); card.setAttribute("aria-label", `${c.name}: ver en grande`);
      card.innerHTML = `<div class="nom"><h3>${c.name}</h3><span class="par n">${dTxt(c)}</span></div>
        <p class="par" style="margin-top:2px">${c.n} qubits, ${c.k} lógicos</p>
        <div class="lienzo"><canvas aria-hidden="true"></canvas></div>
        <div class="dato"><span class="l">BP decide en</span><div class="cifra" data-n>—</div></div>
        <p class="chico" data-t>esperando el primer disparo</p><span class="ir" aria-hidden="true"></span>`;
      cardsEl.appendChild(card);
      const w = new Weave($("canvas", card), {compact: true}); w.setCode(c); weaves.add(w);
      five[c.name] = {card, weave: w, hist: [], ch: null, n: $("[data-n]", card), t: $("[data-t]", card)};
      card.onclick = () => enterOne(c.name); card.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); enterOne(c.name); } };
      const chip = document.createElement("button"); chip.className = "chip"; chip.textContent = c.name; chip.style.setProperty("--c", SER[c.name]); chip.setAttribute("aria-pressed", "true"); shown.add(c.name);
      chip.onclick = () => { if (shown.has(c.name)) { if (shown.size === 1) return; shown.delete(c.name); } else shown.add(c.name); chip.setAttribute("aria-pressed", shown.has(c.name)); syncFive(); };
      chipsEl.appendChild(chip);
    });
  }
  function syncFive() {
    codes.forEach(c => {
      const f = five[c.name], on = shown.has(c.name); f.card.classList.toggle("off", !on);
      if (on && !f.ch && view === "cinco") {
        f.ch = Live.open({code: c.name, mode: "code_capacity", p: curP(), rate_hz: c.name === "BB288" ? 1 : 2, batch: 16});
        f.ch.on("tick", m => onFive(c.name, m)).on("status", refreshStatus); f.ch.connect();
      } else if (!on && f.ch) { f.ch.disconnect(); f.ch = null; f.hist.length = 0; f.n.innerHTML = "—"; f.t.textContent = "apagado"; }
    });
    strip5.series = codes.filter(c => shown.has(c.name)).map(c => ({name: c.name, color: SER[c.name], data: five[c.name].hist, w: 1.8})); strip5.draw();
  }
  function onFive(name, m) {
    const f = five[name]; if (!f.ch || f.ch.serverCode !== name || m.mode !== "code_capacity") return;
    const d = m.display, c = m.counters, p50 = m.latency.amortized_ns.p50;
    num(f.n, p50); f.hist.push(p50); if (f.hist.length > 300) f.hist.shift(); strip5.draw();
    const nErr = d.err_x.length + d.err_z.length, nSyn = d.z_syndrome.length + d.x_syndrome.length;
    f.t.innerHTML = nErr || nSyn ? `Último disparo: <b>${nErr}</b> ${nErr === 1 ? "qubit con falla" : "qubits con falla"}. ${d.logical_fail ? "<b>Quedó una falla lógica.</b>" : "Corregido."}` : "Último disparo sin fallas.";
    f.t.title = `BP resolvió ${loc(c.bp_converged)}, OSD ${loc(c.osd)}, sin solución ${loc(c.failed_decode)}`;
    if (nErr || nSyn) f.weave.show(d);
  }

  /* =====================================================  VISTA DE UNO  ===================================================== */
  const unoEl = $("#uno"), cincoEl = $("#cinco"), W1 = new Weave($("#cv"), {compact: false, tip: $("#tip")}); weaves.add(W1);
  const strip1 = new Strip($("#strip1")); const hist1 = {a: [], s: [], t: []};
  let big = null, mode = "cc", wantMsg = {mode: "code_capacity", decoder: "full"}, roundTimer = null, kind = "interleaved", li = 0, timer = null;
  const TXT = {
    cc: {a: "BP decide cada disparo en", na: "Núcleo BP, 16 disparos por pasada SIMD. <b>Es lo único que baja de 1 µs.</b>", s: "Un disparo solo", t: "Con OSD, el lote entero", L: ["amortizado", "un disparo", "con OSD"]},
    full: {a: "BP sobre el circuito entero", na: "Las 6 rondas como un solo problema. Lo caro viene después de BP: es OSD.", s: "Un disparo solo, BP", t: "Decodificación completa, con OSD", L: ["BP amortizado", "BP un disparo", "completo con OSD"]},
    win: {a: "Cada paso de ventana", na: "Cada ventana resuelve W rondas y confirma F. El costo no crece con la duración.", s: "Por paso, un disparo", t: "Reacción tras la última ronda", L: ["paso (lote)", "paso (uno)", "reacción final"]},
  };
  const tierName = t => t === 0 ? "BP lo resolvió directo" : t === 1 ? "BP no alcanzó: resolvió OSD" : t === 2 ? "Lo resolvió el GNN INT8" : "Sin solución";
  function applyMode(m, send) {
    mode = m; $$("#modos button").forEach(b => b.setAttribute("aria-pressed", b.dataset.m === m)); $("#wf").hidden = m !== "win"; const t = TXT[m];
    $("#l-a").textContent = t.a; $("#l-na").innerHTML = t.na; $("#l-s").textContent = t.s; $("#l-t").textContent = t.t;
    const W = +$("#selW").value, F = Math.min(+$("#selF").value, W - 1); $("#wnote").textContent = m === "win" ? (Math.ceil((7 - W) / F) + 1) + " ventanas" : "";
    wantMsg = m === "cc" ? {mode: "code_capacity", decoder: "full"} : {mode: "circuit", decoder: m === "win" ? "window" : "full"};
    hist1.a.length = hist1.s.length = hist1.t.length = 0; setStrip1(); clearTimeout(roundTimer);
    if (send && big) {
      const cfg = Object.assign({}, wantMsg, m === "cc" ? {rate_hz: 2} : {rate_hz: 1, rounds: 6, schedule: "interleaved", W, F});
      if (m !== "cc" && curP() > .006) { setP((Math.log10(.004) - P0) / (P1 - P0), false); cfg.p = .004; }
      big.ch.configure(cfg);
    }
  }
  function setStrip1() { const t = TXT[mode].L; strip1.series = [{name: t[0], color: C.lino, data: hist1.a, w: 2.2}, {name: t[1], color: C.x, data: hist1.s, w: 1.5}, {name: t[2], color: C.sint, data: hist1.t, w: 1.5}]; strip1.draw(); }
  function playRounds(rounds) { clearTimeout(roundTimer); let i = 0; const step = () => { if (i >= rounds.length) return; W1.show({z_syndrome: rounds[i].z_syndrome, x_syndrome: rounds[i].x_syndrome}); i++; roundTimer = setTimeout(step, 330); }; step(); }
  function onBig(m) {
    if (!big || big.ch.serverCode !== big.name) return;
    if (m.mode !== wantMsg.mode || (m.mode === "circuit" && m.decoder !== wantMsg.decoder)) return;
    const L = m.latency, d = m.display;
    num($("#b-a"), L.amortized_ns.p50); num($("#b-s"), L.bp_kernel_ns.p50); num($("#b-t"), L.total_ns.p50);
    hist1.a.push(L.amortized_ns.p50); hist1.s.push(L.bp_kernel_ns.p50); hist1.t.push(L.total_ns.p50); for (const k in hist1) if (hist1[k].length > 400) hist1[k].shift(); strip1.draw();
    const c = m.counters; $("#c-bp").textContent = loc(c.bp_converged); $("#c-osd").textContent = loc(c.osd); $("#c-gnn").textContent = loc(c.gnn); $("#c-fail").textContent = loc(c.failed_decode);
    const tot = Math.max(1, c.bp_converged + c.osd + c.gnn + c.failed_decode);
    $("#bB").style.width = c.bp_converged / tot * 100 + "%"; $("#bO").style.width = c.osd / tot * 100 + "%"; $("#bG").style.width = c.gnn / tot * 100 + "%"; $("#bF").style.width = c.failed_decode / tot * 100 + "%";
    $("#ler").textContent = m.ler.rate === 0 ? "ninguna todavía" : Live.sci(m.ler.rate);
    $("#lerr").textContent = `${m.ler.failures} fallas en ${loc(c.shots)} disparos. Intervalo de confianza del 95 %: ${Live.sci(m.ler.wilson_lo)} a ${Live.sci(m.ler.wilson_hi)}.`;
    if (m.mode === "circuit") {
      const nDet = d.rounds.reduce((a, r) => a + (r.z_syndrome || []).length + (r.x_syndrome || []).length, 0);
      const how = m.decoder === "window" ? `Lo resolvieron ventanas de ${m.window.W} rondas, confirmando ${m.window.F} por paso.` : "Lo resolvió el horizonte completo.";
      const res = d.logical_fail ? "<b>Quedó una falla lógica.</b>" : "Quedó corregido.";
      $("#last").innerHTML = nDet ? `<b>${nDet}</b> ${nDet === 1 ? "detector disparado" : "detectores disparados"} en <b>${d.rounds.length}</b> ${d.rounds.length === 1 ? "ronda" : "rondas"}. ${how} ${res}` : "Ese lote no tuvo fallas: no hay nada que corregir.";
      if (nDet) playRounds(d.rounds); return;
    }
    const nErr = d.err_x.length + d.err_z.length, nSyn = d.z_syndrome.length + d.x_syndrome.length;
    $("#last").innerHTML = nErr || nSyn ? `<b>${nErr}</b> ${nErr === 1 ? "qubit con falla" : "qubits con falla"} y <b>${nSyn}</b> ${nSyn === 1 ? "chequeo disparado" : "chequeos disparados"}. ${tierName(d.tier_x)}. ${d.logical_fail ? "<b>Quedó una falla lógica.</b>" : "Quedó corregido."}` : "Ese lote no tuvo fallas: no hay nada que corregir.";
    if (nErr || nSyn) W1.show(d);
  }
  function enterOne(name) {
    view = "uno"; bigName = name; const c = by[name];
    Object.values(five).forEach(f => { if (f.ch) { f.ch.disconnect(); f.ch = null; } });
    cincoEl.hidden = true; unoEl.hidden = false; $$("#vista button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === "uno"));
    $("#nom1").textContent = name; $("#par1").textContent = `${c.n} qubits, ${c.k} lógicos, ${c.exact ? "distancia " + c.d_x + " (exacta)" : "distancia entre " + c.d_x_lower + " y " + c.d_x + " (sin certificar)"}`;
    W1.setCode(c); W1.layer = null; hist1.a.length = hist1.s.length = hist1.t.length = 0; $("#last").textContent = "Esperando el primer disparo…";
    const isBig = name === "BB288"; $$("#modos button[data-m=full], #modos button[data-m=win]").forEach(b => { b.disabled = isBig; b.title = isBig ? "BB288: el circuito completo es demasiado pesado para el stream en vivo" : ""; }); if (isBig && mode !== "cc") mode = "cc";
    applyMode(mode, false); setSched();
    if (big) big.ch.disconnect();
    const W = +$("#selW").value, F = Math.min(+$("#selF").value, W - 1);
    big = {name, ch: Live.open(Object.assign({code: name, p: curP(), rate_hz: mode === "cc" ? 2 : 1, batch: 16}, wantMsg, mode === "cc" ? {} : {rounds: 6, schedule: "interleaved", W, F}))};
    big.ch.on("tick", onBig).on("status", refreshStatus); big.ch.connect(); location.hash = "#" + name.toLowerCase(); scrollTo(0, 0);
  }
  function leaveOne() {
    if (big) { big.ch.disconnect(); big = null; }
    view = "cinco"; unoEl.hidden = true; cincoEl.hidden = false; $$("#vista button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === "cinco")); history.replaceState(null, "", location.pathname);
    Object.values(five).forEach(f => f.weave.layout()); strip5.size(); syncFive();
  }
  $("#volver").onclick = leaveOne;
  $$("#vista button").forEach(b => b.onclick = () => { if (b.dataset.v === "cinco") { if (view !== "cinco") leaveOne(); } else if (view === "cinco") enterOne(bigName || "BB72"); });
  $$("#modos button").forEach(b => b.onclick = () => { if (!b.disabled) applyMode(b.dataset.m, true); });
  $("#selW").onchange = () => { const W = +$("#selW").value; [...$("#selF").options].forEach(o => o.disabled = +o.value >= W); if (+$("#selF").value >= W) $("#selF").value = W - 1; applyMode("win", true); };
  $("#selF").onchange = () => applyMode("win", true);

  /* calendario de CNOT */
  const lay = $("#lay");
  function setSched() {
    const code = by[bigName]; if (!code) return; const Ls = code.tanner.layers; $("#tI").setAttribute("aria-pressed", kind === "interleaved"); $("#tS").setAttribute("aria-pressed", kind === "sequential");
    const layers = Ls[kind]; if (!layers) { W1.layer = null; $("#ltxt").textContent = "Este calendario todavía no está calculado para este código."; return; }
    lay.max = layers.length - 1; li = Math.min(li, layers.length - 1); lay.value = li; W1.layer = layers[li]; W1.dirty = true;
    const x = layers[li].filter(e => e[0] === "X").length;
    $("#lcount").textContent = `capa ${li + 1} de ${layers.length}`;
    $("#ltxt").innerHTML = `<b>${layers[li].length}</b> CNOT a la vez (${x} de chequeos X, ${layers[li].length - x} de Z). ${kind === "interleaved" ? "Mezcla X y Z en la misma capa." : "Primero todos los X, después todos los Z."} La profundidad pasa de ${code.depth.sequential} a ${code.depth.interleaved} capas.`;
  }
  $("#tI").onclick = () => { kind = "interleaved"; setSched(); }; $("#tS").onclick = () => { kind = "sequential"; setSched(); }; lay.oninput = () => { li = +lay.value; setSched(); };
  $("#tP").onclick = () => { W1.playing = !W1.playing; $("#tP").setAttribute("aria-pressed", W1.playing); $("#tP").textContent = W1.playing ? "Pausar" : "Reproducir"; clearInterval(timer); if (W1.playing) timer = setInterval(() => { const n = (by[bigName].tanner.layers[kind] || []).length; li = (li + 1) % n; setSched(); }, 900); };

  /* Emblema: el mismo anillo de la web, con clavos y cuerdas entre ellos. */
  function marca(cv) {
    const w = 46, d = Math.min(devicePixelRatio || 1, 2); cv.width = w * d; cv.height = w * d;
    const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, w, w);
    const N = 18, R = 19, c = w / 2, P = k => [c + Math.cos(-Math.PI / 2 + k / N * 2 * Math.PI) * R, c + Math.sin(-Math.PI / 2 + k / N * 2 * Math.PI) * R];
    g.lineWidth = .7; g.strokeStyle = "rgba(169,189,255,.75)"; g.beginPath(); for (let k = 0; k < N; k++) { const a = P(k), b = P((k + 6) % N); g.moveTo(...a); g.lineTo(...b); } g.stroke();
    g.strokeStyle = "rgba(255,141,160,.75)"; g.beginPath(); for (let k = 0; k < N; k++) { const a = P(k), b = P((k + 8) % N); g.moveTo(...a); g.lineTo(...b); } g.stroke();
    g.fillStyle = "#f6bd44"; g.beginPath(); g.arc(c, c, 2.4, 0, 7); g.fill();
    g.fillStyle = "#f1ead6"; for (let k = 0; k < N; k++) { const [x, y] = P(k); g.beginPath(); g.arc(x, y, 1.5, 0, 7); g.fill(); }
  }
  marca($("#marca"));

  /* =====================================================  ARRANQUE  ===================================================== */
  makeRail($("#riel5"), "Probabilidad de error físico"); makeRail($("#riel1"), "Probabilidad de error físico"); setP(pf);
  function refreshStatus() {
    const chs = [...Object.values(five).map(f => f.ch), big && big.ch].filter(Boolean), on = chs.some(c => c.status === "on");
    $("#punto").className = "punto" + (on ? " on" : "");
    $("#conexion").textContent = on ? "Decodificando en esta máquina" : chs.length ? "Conectando con el decodificador…" : "Sin señal del decodificador";
  }
  Live.codes().then(cs => {
    codes = cs; cs.forEach(c => by[c.name] = c); buildCards(); syncFive(); refreshStatus();
    const h = location.hash.slice(1).toUpperCase(); if (by[h]) enterOne(h);
    seccionDepth(); seccionPrueba();
  }).catch(e => { console.error("Tars: falló el arranque", e); $("#conexion").textContent = "Sin señal del decodificador"; });

  /* =====================================================  MÁS ABAJO  ===================================================== */
  const NS = "http://www.w3.org/2000/svg", se = (n, a, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); p && p.appendChild(e); return e; };
  function seccionDepth() {
    const box = $("#prof"), c0 = codes[0], seq = c0.depth.sequential, ent = c0.depth.interleaved;
    const fila = (n, col, txt) => `<div class="capas"><div class="seg">${Array.from({length: n}, () => `<i style="background:${col}"></i>`).join("")}</div><p>${txt}</p></div>`;
    const igual = codes.every(c => c.depth.sequential === seq && c.depth.interleaved === ent);
    box.innerHTML = fila(seq, "rgba(195,193,212,.55)", `<b>${seq} capas</b> en secuencia`) + fila(ent, C.sint, `<b>${ent} capas</b> entrelazado`) +
      (igual ? `<p class="nota">Vale para los cinco códigos.</p>` : "") +
      `<div class="libro">` + codes.map(c => `<span>${c.name}</span><span class="n">${loc(c.cnot_per_round)} CNOT por ronda</span>`).join("") + `</div>`;
  }
  const plotOn = new Set(["BB72", "BB90", "BB108", "BB144"]); let plotData = null;
  const PCOL = {BB72: C.lino, BB90: C.x, BB108: C.sint, BB144: C.z};
  function drawPlot() {
    if (!plotData) return; const {I, S} = plotData, svg = $("#plot"), W = 700, H = 470, M = {l: 58, r: 14, t: 16, b: 46}; svg.innerHTML = "";
    const px = p => M.l + (Math.log10(p) - Math.log10(4e-4)) / (Math.log10(8e-3) - Math.log10(4e-4)) * (W - M.l - M.r), py = v => M.t + (0 - Math.log10(v)) / 5 * (H - M.t - M.b);
    const tx = (x, y, t, a) => { const e = se("text", {x, y, "text-anchor": a, style: "font:400 12px Martian;fill:" + C.lino2}, svg); e.textContent = t; };
    for (let e = 0; e >= -5; e--) { const y = py(Math.pow(10, e)); se("line", {x1: M.l, x2: W - M.r, y1: y, y2: y, stroke: C.lino, opacity: .14}, svg); tx(M.l - 8, y + 4, "1e" + e, "end"); }
    [5e-4, 1e-3, 2e-3, 4e-3].forEach(p => tx(px(p), H - 18, Live.sci(p), "middle"));
    se("line", {x1: px(5e-4), y1: py(5e-4), x2: px(8e-3), y2: py(8e-3), stroke: C.lino, "stroke-dasharray": "3 5", opacity: .45}, svg);
    const draw = (run, dash) => ["BB72", "BB90", "BB108", "BB144"].forEach(cn => { if (!plotOn.has(cn)) return; const pts = run.points.filter(p => p.code === cn && p.p_logical > 0).sort((a, b) => a.p - b.p); if (pts.length < 2) return;
      se("path", {d: pts.map((p, i) => (i ? "L" : "M") + px(p.p).toFixed(1) + " " + py(p.p_logical).toFixed(1)).join(" "), fill: "none", stroke: PCOL[cn], "stroke-width": 2.2, "stroke-dasharray": dash || "none"}, svg);
      pts.forEach(p => { const c = se("circle", {cx: px(p.p), cy: py(p.p_logical), r: 3.4, fill: dash ? C.fondo : PCOL[cn], stroke: PCOL[cn], "stroke-width": 1.4}, svg); se("title", {}, c).textContent = `${cn}, ${dash ? "12" : "7"} capas, p=${Live.sci(p.p)}: ${Live.sci(p.p_logical)} (${p.failures} fallas en ${p.shots} disparos)`; }); });
    draw(S, "5 5"); draw(I, null);
  }
  function seccionPrueba() {
    Live.bench().then(b => {
      const I = b.runs.final_int, S = b.runs.final_seq; plotData = {I, S};
      $("#p-th").innerHTML = `Umbral, donde se cruzan las curvas de distinto tamaño: <b>${Live.sci(S.threshold.estimate)}</b> con 12 capas y <b>${Live.sci(I.threshold.estimate)}</b> con 7. Medido en ${loc(I.total_shots + S.total_shots)} disparos de circuito completo.`;
      const ch = $("#chips-plot"); ch.innerHTML = ""; ["BB72", "BB90", "BB108", "BB144"].forEach(n => { const b = document.createElement("button"); b.className = "chip"; b.textContent = n; b.style.setProperty("--c", PCOL[n]); b.setAttribute("aria-pressed", "true"); b.onclick = () => { if (plotOn.has(n)) { if (plotOn.size === 1) return; plotOn.delete(n); } else plotOn.add(n); b.setAttribute("aria-pressed", plotOn.has(n)); drawPlot(); }; ch.appendChild(b); });
      drawPlot();
      $("#honest").innerHTML = '<p class="tit">Tars</p><p>Los números salen de las ejecuciones guardadas y del decodificador que corre ahora mismo en esta máquina, un procesador de escritorio con AVX2. <b>Lo único que baja de 1 µs es el núcleo BP, por disparo, en lotes de 16 y sobre la matriz de una ronda.</b> El decodificador completo con OSD sobre el circuito entero tarda cientos de microsegundos a milisegundos. Cero fallas observadas significa cero en los disparos que se corrieron, no cero verdadero. Todo es simulación; no se corrió nada en hardware cuántico.</p>';
    }).catch(() => {});
  }
  fetch("/api/v1/compile", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({code: "BB72", schedule: "interleaved", rounds: 3, p: 1e-3})}).then(r => r.json()).then(j => {
    $("#q-t").textContent = `BB72, tres rondas: ${j.stats.qasm_lines} líneas de OpenQASM 3.0, ${j.stats.cnot_layers} capas de CNOT por ronda, ${j.stats.cnot_total} CNOT en total. Compilado ahora, en ${Math.round(j.elapsed_ms)} ms.`; $("#qasm").textContent = j.qasm; }).catch(e => { $("#q-t").textContent = "No pude compilar el circuito: " + e.message; });

  /* =====================================================  DEL CÓDIGO AL SILICIO  ===================================================== */
  /* Todo lo de abajo sale de /api/v1/hardware (resultados guardados) y de /api/v1/rtl/vector (el modelo bit a bit, comparado en vivo con el núcleo C++). */
  const fx = (v, d = 1) => nf(v, d), txt = (x, y, t, a, svg, col) => { const e = se("text", {x, y, "text-anchor": a || "start", style: "font:400 12px Martian;fill:" + (col || C.lino2)}, svg); e.textContent = t; return e; };
  const chip = (box, label, on, fn) => { const b = document.createElement("button"); b.className = "chip"; b.textContent = label; b.setAttribute("aria-pressed", on); b.onclick = () => { $$(".chip", box).forEach(x => x.setAttribute("aria-pressed", x === b)); fn(); }; box.appendChild(b); return b; };

  function seccionDistancia(h) {
    const runs = h.circuit_distance.filter(r => r.code === "BB72" && r.rounds === 6 && r.basis === "Z");
    const seq = runs.find(r => r.schedule === "sequential"), ent = runs.find(r => r.schedule === "interleaved");
    if (!seq || !ent) { $("#d-t").textContent = "Todavía no hay mediciones guardadas de la distancia de circuito."; return; }
    const dc = seq.distance_code, dec = (h.circuit_distance_decision || []).find(r => r.code === "BB72" && r.schedule === "interleaved" && r.basis !== "X");
    let t = `BB72 tiene distancia ${dc} como código. Con ruido en cada puerta del circuito real, 6 rondas, la distancia efectiva es menor: con ${seq.cnot_layers} capas de CNOT vale <b>${seq.d_lower}</b>, demostrado por el solver en ${loc(Math.round(seq.seconds))} s. Con ${ent.cnot_layers} capas el solver la acotó entre <b>${ent.d_lower}</b> y <b>${ent.d_upper}</b>, y no pudo cerrarla en ${loc(ent.time_limit)} s.`;
    let lo = ent.d_lower, hi = ent.d_upper;
    if (dec) {
      if (dec.proved_no_fault_up_to != null) { lo = Math.max(lo, dec.d_lower); t += ` Después probó que no existe ninguna falla no detectada de peso ${dec.max_weight} o menos.` + (lo === hi ? ` Con eso, la distancia de las ${ent.cnot_layers} capas es <b>exactamente ${hi}</b>.` : ""); }
      else if (dec.d_upper != null) { hi = Math.min(hi, dec.d_upper); t += ` Preguntando solo si existe una falla de peso ${dec.max_weight} o menos, halló una de peso ${dec.d_upper}.`; }
      else t += ` Preguntarle solo si existe una falla de peso ${dec.max_weight} o menos tampoco terminó en ${loc(dec.time_limit)} s.`;
    }
    const xe = h.circuit_distance.find(r => r.code === "BB72" && r.schedule === "interleaved" && r.rounds === 6 && r.basis === "X"), xd = (h.circuit_distance_decision || []).find(r => r.code === "BB72" && r.schedule === "interleaved" && r.basis === "X");
    if (xe && xd && xd.proved_no_fault_up_to != null && xd.d_lower === xe.d_upper) t += ` En la base X pasa lo mismo: también es <b>exactamente ${xe.d_upper}</b>.`;
    $("#d-t").innerHTML = t;
    $("#d-n").innerHTML = "Una falla de peso <b>w</b> son w errores físicos del circuito que ningún detector ve y que, juntos, dan vuelta un qubit lógico. La distancia de circuito es el menor w posible. Cuenta los errores que se propagan por las puertas, que la distancia del código no ve. Una cota sin certificar es lo que hay, no una conclusión.";
    const svg = $("#d-plot"); svg.innerHTML = ""; const W = 700, M = {l: 120, r: 24}, top = 40, dmax = Math.max(dc, hi, seq.d_upper) + 1, X = d => M.l + d / dmax * (W - M.l - M.r);
    for (let d = 0; d <= dmax; d++) { se("line", {x1: X(d), x2: X(d), y1: top - 8, y2: 200, stroke: C.lino, opacity: .12}, svg); txt(X(d), 222, String(d), "middle", svg); }
    se("line", {x1: X(dc), x2: X(dc), y1: top - 14, y2: 200, stroke: C.sint, "stroke-dasharray": "4 5"}, svg); txt(X(dc), top - 20, `distancia del código: ${dc}`, "middle", svg, C.sint);
    const fila = (y, nombre, lo_, hi_, cert) => {
      txt(M.l - 12, y + 5, nombre, "end", svg, C.lino);
      if (hi_ == null) hi_ = dmax;
      const pt = hi_ === lo_;
      se("rect", {x: pt ? X(lo_) - 11 : X(lo_), y: y - 11, width: pt ? 22 : Math.max(2, X(hi_) - X(lo_)), height: 22, fill: cert ? C.ok : C.x, opacity: cert ? .95 : .35, stroke: cert ? C.ok : C.x, "stroke-dasharray": cert ? "" : "4 4"}, svg);
      txt((X(lo_) + X(hi_)) / 2, y + 5, cert || lo_ === hi_ ? String(lo_) : `${lo_} a ${hi_}`, "middle", svg, cert ? C.fondo : C.lino);
    };
    fila(90, `${seq.cnot_layers} capas`, seq.d_lower, seq.d_upper, seq.certified);
    fila(150, `${ent.cnot_layers} capas`, lo, hi, lo === hi);
    svg.setAttribute("aria-label", `Distancia de circuito de BB72: ${seq.cnot_layers} capas ${seq.d_lower}, certificada; ${ent.cnot_layers} capas entre ${lo} y ${hi}`);
    const lib = $("#d-lib"); lib.innerHTML = "";
    const otros = h.circuit_distance.filter(r => r.code !== "BB72").sort((a, b) => a.code.localeCompare(b.code, "es", {numeric: true}) || b.cnot_layers - a.cnot_layers);
    otros.forEach(r => lib.insertAdjacentHTML("beforeend", `<span>${r.code} · ${r.cnot_layers} capas · ${r.rounds} rondas</span><span class="n">${r.certified ? "exactamente " + r.d_lower : "entre " + r.d_lower + (r.d_upper != null ? " y " + r.d_upper : " y ?")} · ${loc(Math.round(r.seconds))} s</span>`));
    if (otros.length) lib.insertAdjacentHTML("afterbegin", `<span><b>Códigos más grandes</b>, con tiempo acotado</span><span class="n">distancia del código ${[...new Set(otros.map(r => r.code + " " + r.distance_code))].join(", ")}</span>`);
  }

  function seccionCompuertas(h) {
    const G = h.logical_gates, names = Object.keys(G); if (!names.length) { $("#g-t").textContent = "Todavía no hay compuertas medidas."; return; }
    const chips = $("#g-chips");
    function show(n) {
      const g = G[n], ex = g.examples[0], k = g.k, mem = g.in_memory_circuit;
      $("#g-t").innerHTML = `Correr todos los qubits de ${n} un paso por el toro de ${g.l} por ${g.m} no cambia ninguno de sus chequeos, pero sí le hace algo a los ${k} qubits lógicos: una compuerta que se logra renombrando qubits, sin una sola puerta física. De las ${g.automorphisms} traslaciones, ${g.nontrivial_logical_gates} actúan de forma no trivial, con ${g.distinct_logical_matrices} acciones distintas. En la matriz, cada fila dice en qué qubits lógicos se convierte uno: el desplazamiento (${ex.shift.join(", ")}) de ${n}.`;
      const box = $("#g-u"); box.innerHTML = ""; box.style.gridTemplateColumns = `repeat(${k}, 1fr)`; box.setAttribute("aria-label", `Matriz lógica de ${k} por ${k} de la traslación (${ex.shift.join(", ")}) de ${n}; ${ex.u_z_ones} unos`);
      ex.u_z.forEach(row => row.forEach(v => box.appendChild(el("i", v ? "u" : "", ""))));
      $("#g-n").innerHTML = `Stim, el simulador de estabilizadores, contrastó ${loc(g.stim_predictions_checked)} predicciones de cómo cada traslación mueve los operadores lógicos, con <b>${g.stim_mismatches} discrepancias</b>, y confirmó que el espacio del código se conserva. Metidas una por una entre rondas del calendario de ${mem.schedule_layers} capas: <b>${mem.gates_inserted} compuertas, ${mem.violations_total} violaciones, ${mem.gauge_detectors} detectores de gauge</b>.`;
      const lib = $("#g-lib"); lib.innerHTML = "";
      [["Traslaciones probadas", loc(g.translations_tested)], ["Son simetrías del código", loc(g.automorphisms)], ["Compuertas lógicas no triviales", loc(g.nontrivial_logical_gates)], ["Acciones lógicas distintas", loc(g.distinct_logical_matrices)], ["Unos en la matriz mostrada", `${ex.u_z_ones} de ${k * k}`]].forEach(([a, b]) => lib.insertAdjacentHTML("beforeend", `<span>${a}</span><span class="n">${b}</span>`));
    }
    names.forEach((n, i) => chip(chips, n, i === 0, () => show(n))); show(names[0]);
  }

  function seccionVentana(h) {
    const runs = h.window_fused; if (!runs.length) { $("#w-t").textContent = "Todavía no hay mediciones de la ventana fusionada."; return; }
    const opts = []; runs.forEach(r => Object.values(r.configs).forEach(c => opts.push({r, c})));
    const chips = $("#w-chips"), body = $("#w-body");
    function show({r, c}) {
      const py = c.python_loop, fu = c.fused_cpp, sp = fu.split_batch_per_shot_us, tot = sp.bp + sp.post + sp.prop, red = 100 * (1 - fu.wall_per_shot_us / py.wall_per_shot_us);
      $("#w-t").innerHTML = `La ventana deslizante decide por tramos: resuelve las primeras rondas, confirma ${c.F} y arrastra su efecto a la ventana siguiente. Todo ese bucle corre ahora dentro del decodificador en C++, en una sola llamada. En ${r.code}, ventana (${c.W}, ${c.F}), tarda <b>${fx(fu.wall_per_shot_us, 0)} µs por disparo</b> contra ${fx(py.wall_per_shot_us, 0)} µs con el bucle en Python: <b>${fx(red, 0)} % menos</b>. Las decisiones son ${c.identical_decisions ? "<b>idénticas</b>, disparo por disparo" : "DISTINTAS"} (${loc(r.shots)} disparos, semilla ${r.seed}).`;
      const bar = (a, col) => `<div class="tira"><i style="width:${a}%;background:${col}"></i></div>`;
      const pct = x => (100 * x / tot).toFixed(1);
      body.innerHTML = `<div class="fila"><div class="et"><span>Bucle en Python</span><b class="n">${fx(py.wall_per_shot_us, 0)} µs por disparo</b></div>${bar(100, "rgba(195,193,212,.55)")}</div>
        <div class="fila"><div class="et"><span>Bucle en C++, una llamada</span><b class="n">${fx(fu.wall_per_shot_us, 0)} µs por disparo</b></div>${bar(100 * fu.wall_per_shot_us / py.wall_per_shot_us, C.sint)}</div>
        <div class="fila"><div class="et"><span>En qué se va cada paso de ventana, por disparo</span><b class="n">${fx(tot, 1)} µs</b></div>
          <div class="tira"><i style="width:${pct(sp.bp)}%;background:${C.x}"></i><i style="width:${pct(sp.post)}%;background:${C.sint}"></i><i style="width:${pct(sp.prop)}%;background:${C.z}"></i></div>
          <div class="leyenda"><span><i style="background:${C.x}"></i>BP ${fx(sp.bp, 1)} µs</span><span><i style="background:${C.sint}"></i>OSD ${fx(sp.post, 1)} µs</span><span><i style="background:${C.z}"></i>armar y propagar ${fx(sp.prop, 1)} µs</span></div></div>`;
      $("#w-n").innerHTML = `Un disparo suelto no se acelera: lo domina el propio BP (mediana ${fx(fu.step_single_us.p50_us, 1)} µs por paso con el bucle fusionado, ${fx(py.step_single_us.p50_us, 1)} µs con Python). Lo que se gana es rendimiento sostenido. El OSD pesa tanto porque corre solo en los disparos donde BP no converge. Mejor de ${loc(r.wall_time_best_of || 1)} corridas, ruido ${Live.sci(r.p)}, ${r.rounds} rondas.`;
    }
    opts.forEach((o, i) => chip(chips, `${o.r.code} · ${o.c.W},${o.c.F}`, i === 0, () => show(o))); show(opts[0]);
  }

  function seccionSilicio(h) {
    const R = h.rtl; if (!R) { $("#s-t").textContent = "Todavía no se generó el RTL."; return; }
    const cy = R.cycles, c250 = R.nominal_clock_conversion_us["250MHz"], c400 = R.nominal_clock_conversion_us["400MHz"], syn = R.synthesis, reg = syn && syn.registers.total;
    $("#s-t").innerHTML = `El núcleo BP de ${R.code} (matriz de ${R.checks} por ${R.variables}) escrito como SystemVerilog sintetizable: números enteros de 16 bits, sumas que saturan, un chequeo por nodo y todos a la vez. El archivo se genera desde el mismo grafo que usa el decodificador. Se ejecutó <b>ciclo a ciclo</b> con ${loc(R.shots)} síndromes con ruido y coincide <b>bit a bit</b> con el C++: los ${R.variables} valores finales (LLR de 16 bits) de cada disparo, las decisiones y el número de iteraciones.`;
    $("#s-n").innerHTML = `Son ciclos de reloj del RTL, no tiempo de CPU. Las frecuencias de 250 y 400 MHz son nominales, solo para convertir: <b>no se sintetizó ni se probó en ninguna placa</b>. Lo que se midió es BP sobre la matriz de una ronda, no el decodificador completo con OSD. El banco de pruebas ${R.testbench_executed ? "se ejecutó" : "se genera pero no se ejecutó aquí"}: no hay simulador Verilog en esta máquina; el RTL se corrió traducido a C++ con Yosys.`;
    const dl = $("#s-dl"); dl.innerHTML = "";
    h.rtl_files.forEach(f => { const a = document.createElement("a"); a.className = "chip"; a.href = "/api/v1/rtl/file/" + f; a.setAttribute("download", f); a.textContent = f.startsWith("tb_") ? "Banco de pruebas" : f.endsWith("_in.mem") ? "Síndromes (.mem)" : f.endsWith("_out.mem") ? "Resultados esperados (.mem)" : "Descargar el .sv"; dl.appendChild(a); });
    const lib = $("#s-lib"); lib.innerHTML = "";
    [["Coinciden con el C++", `${loc(R.bit_exact.rtl_vs_cpp.vectors)} de ${loc(R.shots)} síndromes`], ["Ciclos por síndrome: mediana, p95, máximo", `${Math.round(cy.p50)} · ${Math.round(cy.p95)} · ${cy.max}`], ["Fórmula de ciclos", `${cy.formula}, exacta en todos`],
     ["A 250 MHz nominales: mediana, máximo", `${fx(c250.p50, 3)} · ${fx(c250.max, 3)} µs`], ["A 400 MHz nominales: mediana, máximo", `${fx(c400.p50, 3)} · ${fx(c400.max, 3)} µs`],
     ["Bits de registro", reg ? loc(reg) : "—"], ["Compuertas genéricas, cota superior", syn ? loc(syn.combinational_gates_total) : "—"]]
      .forEach(([a, b]) => lib.insertAdjacentHTML("beforeend", `<span>${a}</span><span class="n">${b}</span>`));
    const F = h.rtl_formal;
    if (F) lib.insertAdjacentHTML("afterbegin", `<span><b>Demostrado para todas las entradas</b></span><span>${F.all_proved ? "Nodo de chequeo de grado " + F.checks[0].degree + " (" + loc(F.checks[0].inputs_bits) + " bits de entrada), nodos de variable y prueba de parada: equivalentes a la especificación del C++, con un solver (z3). Probado en " + fx(F.checks[0].equivalence_seconds, 1) + " s. No cubre la secuencia de registros ni el cableado entre bloques." : "Hay un bloque que NO se pudo demostrar."}</span>`);
    if (syn) lib.insertAdjacentHTML("beforeend", `<span>Qué es ese número</span><span>Síntesis genérica de Yosys sin optimizar; no es un conteo de LUT ni de ninguna placa.</span>`);
    const sl = $("#s-i"), ex = $("#s-ex"); ex.innerHTML = ""; $("#s-nv").textContent = loc(R.shots); sl.max = R.shots - 1;
    const E = h.rtl_examples || {}, go = i => { sl.value = i; load(); };
    [["no_iterations", "Sin ruido que corregir"], ["typical", "Uno típico"], ["slowest_converging", "El más lento que converge"], ["not_converged", "Uno que no converge"]].forEach(([k, l]) => { if (E[k] != null) chip(ex, l, false, () => go(E[k])); });
    let tm = 0, seq = 0;
    function load() {
      const i = +sl.value, my = ++seq; $("#s-iv").textContent = loc(i); sl.setAttribute("aria-valuetext", `síndrome ${i}`);
      fetch("/api/v1/rtl/vector/" + i).then(r => r.json()).then(v => { if (my === seq) draw(v); }).catch(() => { $("#s-lec").textContent = "No se pudo leer ese síndrome."; });
    }
    function draw(v) {
      $("#s-lec").innerHTML = `${v.syndrome_weight} chequeos encendidos por ${v.error_weight} errores. ${v.converged ? `Convergió en <b>${v.iterations}</b> iteraciones` : `<b>No convergió</b> tras ${v.iterations} iteraciones, con ${v.residual_syndrome_weight} chequeos sin satisfacer`}: <b>${v.cycles} ciclos</b>, ${fx(v.us_at["250MHz"], 3)} µs a 250 MHz nominales. ${v.identical_to_cpp_kernel ? "El núcleo C++ da exactamente lo mismo." : "<b>El núcleo C++ NO da lo mismo.</b>"}`;
      const svg = $("#s-tr"); svg.innerHTML = ""; const W = 700, M = {l: 46, r: 12, t: 22, b: 40}, u = v.unsatisfied_per_iteration, n = u.length, mx = Math.max(1, ...u), bw = (W - M.l - M.r) / Math.max(n, 6);
      const Y = c => 250 - M.b - c / mx * (250 - M.b - M.t - 8);
      se("line", {x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0), stroke: C.lino, opacity: .4}, svg);
      txt(M.l - 8, Y(mx) + 4, String(mx), "end", svg); txt(M.l - 8, Y(0) + 4, "0", "end", svg); txt(M.l, 14, "chequeos sin satisfacer", "start", svg);
      u.forEach((c, k) => { const x = M.l + k * bw + bw * .12, last = k === n - 1;
        se("rect", {x, y: Math.min(Y(c), Y(0) - (c || !last ? 0 : 3)), width: bw * .76, height: Math.max(c || last ? 3 : 0, Y(0) - Y(c)), fill: last ? (v.converged ? C.ok : C.mal) : C.x, opacity: last ? 1 : .8}, svg);
        if (n <= 16 || k % 5 === 0 || last) txt(x + bw * .38, 250 - M.b + 16, String(k), "middle", svg); });
      txt((M.l + W - M.r) / 2, 246, "iteración", "middle", svg);
      svg.setAttribute("aria-label", `Chequeos sin satisfacer en cada iteración: ${u.join(", ")}`);
    }
    sl.addEventListener("input", () => { $("#s-iv").textContent = loc(+sl.value); clearTimeout(tm); tm = setTimeout(load, 90); });
    load();
  }

  Live.hardware().then(h => { seccionDistancia(h); seccionCompuertas(h); seccionVentana(h); seccionSilicio(h);
    $("#honest").insertAdjacentHTML("beforeend", "<p>Del código al silicio: la distancia de circuito, las compuertas y la ventana fusionada son mediciones guardadas. El RTL se ejecutó ciclo a ciclo pero <b>no se sintetizó para ninguna placa</b>, así que no hay frecuencia ni consumo medidos. Lo que se dice de las frecuencias de 250 y 400 MHz es una conversión nominal.</p>"); }).catch(e => console.error("Tars: falló la sección de hardware", e));
})();
