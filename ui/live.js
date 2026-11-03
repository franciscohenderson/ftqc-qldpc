/* Capa de datos en vivo. Nada acá es simulado: cada número sale del decodificador C++ (WS /api/v1/stream/decode)
   o de los resultados guardados.
   Live.open(cfg) abre un canal propio (uno por código en la vista de los cinco); la API de siempre
   (Live.on / connect / configure / disconnect) sigue funcionando sobre un canal por defecto. */
"use strict";
window.Live = (() => {
  const nf = (n, d = 2) => Number(n).toLocaleString("es-AR", {minimumFractionDigits: d, maximumFractionDigits: d});
  const sci = v => { if (v == null) return "—"; if (v === 0) return "0"; const [m, e] = v.toExponential(1).split("e"); return m.replace(".", ",") + "e" + e.replace("+", "").replace("-0", "-"); };
  /* ns -> {v:"0,27", u:"µs"} */
  const t = ns => { if (ns == null || isNaN(ns)) return {v: "—", u: ""}; const a = Math.abs(ns);
    if (a < 1000) return {v: String(Math.round(ns)), u: "ns"};
    if (a < 1e6) return {v: nf(ns / 1e3, a < 1e4 ? 2 : 1), u: "µs"};
    return {v: nf(ns / 1e6, 2), u: "ms"}; };

  const DEFAULTS = {code: "BB72", mode: "code_capacity", p: 0.01, rate_hz: 8, batch: 16, schedule: "interleaved", rounds: 3, decoder: "full", W: 3, F: 1};

  function channel(init) {
    const H = {tick: [], hello: [], error: [], status: []};
    let ws = null, cfg = Object.assign({}, DEFAULTS, init), want = false, retry = 0, timer = null, status = "idle";
    const ch = {last: null, serverCode: null};
    const emit = (k, ...a) => H[k].forEach(f => { try { f(...a); } catch (e) { console.error(e); } });
    const st = s => { status = s; emit("status", s); };

    function open() {
      clearTimeout(timer);
      try { ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/v1/stream/decode"); }
      catch (e) { st("error"); return; }
      st("connecting");
      const mine = ws;
      mine.onopen = () => { retry = 0; st("on"); mine.send(JSON.stringify(cfg)); };
      mine.onmessage = ev => {
        let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
        if (m.type === "tick") { ch.last = m; emit("tick", m); }
        else if (m.type === "hello") { ch.serverCode = m.config && m.config.code; emit("hello", m); }
        else if (m.type === "error") emit("error", m.message);
      };
      mine.onclose = () => { if (ws === mine) st("off"); if (want && ws === mine) { timer = setTimeout(open, Math.min(5000, 800 * ++retry)); } };
      mine.onerror = () => { st("error"); };
    }
    Object.assign(ch, {
      on(k, f) { H[k].push(f); return ch; },
      off(k, f) { H[k] = H[k].filter(x => x !== f); return ch; },
      connect(c) { cfg = Object.assign(cfg, c || {}); want = true; if (!ws || ws.readyState > 1) open(); else if (ws.readyState === 1) ws.send(JSON.stringify(cfg)); return ch; },
      configure(c) { cfg = Object.assign(cfg, c); if (ws && ws.readyState === 1) ws.send(JSON.stringify(cfg)); return ch; },
      disconnect() { want = false; clearTimeout(timer); if (ws) { const w = ws; ws = null; w.close(); st("off"); } return ch; },
    });
    /* getters vivos (Object.assign copiaría el valor una sola vez) */
    Object.defineProperties(ch, {status: {get: () => status}, cfg: {get: () => cfg}});
    return ch;
  }

  const memo = {};
  const get = (k, url) => memo[k] || (memo[k] = fetch(url).then(r => { if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }));
  const main = channel();
  const api = {nf, sci, t, open: channel,
    on: (k, f) => { main.on(k, f); return api; }, off: (k, f) => { main.off(k, f); return api; },
    connect: c => { main.connect(c); return api; }, configure: c => { main.configure(c); return api; }, disconnect: () => { main.disconnect(); return api; },
    get last() { return main.last; }, get status() { return main.status; }, get cfg() { return main.cfg; },
    codes: () => get("codes", "/api/v1/codes").then(j => j.codes),
    code: name => api.codes().then(cs => cs.find(c => c.name === name)),
    bench: () => get("bench", "/api/v1/benchmarks"),
    hardware: () => get("hw", "/api/v1/hardware"),
    rm: matchMedia("(prefers-reduced-motion: reduce)").matches};
  return api;
})();
