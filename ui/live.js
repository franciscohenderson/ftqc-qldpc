/* Capa de datos en vivo, compartida por las tres direcciones. Nada acá es simulado:
   cada número sale del decodificador C++ (WS /api/v1/stream/decode) o de los resultados guardados. */
"use strict";
window.Live = (() => {
  const nf = (n, d = 2) => Number(n).toLocaleString("es-AR", {minimumFractionDigits: d, maximumFractionDigits: d});
  const sci = v => { if (v == null) return "—"; if (v === 0) return "0"; const [m, e] = v.toExponential(1).split("e"); return m.replace(".", ",") + "e" + e.replace("+", "").replace("-0", "-"); };
  /* ns -> {v:"0,27", u:"µs"} */
  const t = ns => { if (ns == null || isNaN(ns)) return {v: "—", u: ""}; const a = Math.abs(ns);
    if (a < 1000) return {v: String(Math.round(ns)), u: "ns"};
    if (a < 1e6) return {v: nf(ns / 1e3, a < 1e4 ? 2 : 1), u: "µs"};
    return {v: nf(ns / 1e6, 2), u: "ms"}; };

  const H = {tick: [], hello: [], error: [], status: []};
  const on = (k, f) => { H[k].push(f); return api; };
  const off = (k, f) => { H[k] = H[k].filter(x => x !== f); return api; };
  const emit = (k, ...a) => H[k].forEach(f => { try { f(...a); } catch (e) { console.error(e); } });

  let ws = null, cfg = {code: "BB72", mode: "code_capacity", p: 0.01, rate_hz: 8, batch: 16, schedule: "interleaved", rounds: 3, decoder: "full", W: 3, F: 1};
  let want = false, retry = 0, timer = null, status = "idle";
  const st = s => { status = s; emit("status", s); };

  function open() {
    clearTimeout(timer);
    try { ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/v1/stream/decode"); }
    catch (e) { st("error"); return; }
    st("connecting");
    ws.onopen = () => { retry = 0; st("on"); ws.send(JSON.stringify(cfg)); };
    ws.onmessage = ev => {
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (m.type === "tick") { api.last = m; emit("tick", m); }
      else if (m.type === "hello") emit("hello", m);
      else if (m.type === "error") emit("error", m.message);
    };
    ws.onclose = () => { st("off"); if (want) { timer = setTimeout(open, Math.min(5000, 800 * ++retry)); } };
    ws.onerror = () => { st("error"); };
  }
  function connect(c) { cfg = Object.assign(cfg, c || {}); want = true; if (!ws || ws.readyState > 1) open(); else if (ws.readyState === 1) ws.send(JSON.stringify(cfg)); return api; }
  function configure(c) { cfg = Object.assign(cfg, c); if (ws && ws.readyState === 1) ws.send(JSON.stringify(cfg)); }
  function disconnect() { want = false; if (ws) ws.close(); }

  const memo = {};
  const get = (k, url) => memo[k] || (memo[k] = fetch(url).then(r => { if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }));
  const api = {nf, sci, t, on, off, connect, configure, disconnect, last: null,
    get status() { return status; }, get cfg() { return cfg; },
    codes: () => get("codes", "/api/v1/codes").then(j => j.codes),
    code: name => api.codes().then(cs => cs.find(c => c.name === name)),
    bench: () => get("bench", "/api/v1/benchmarks"),
    rm: matchMedia("(prefers-reduced-motion: reduce)").matches};
  return api;
})();
