// Capture real screenshots of the live console for the README.  Usage:
//   node scripts/capture_assets.mjs [http://localhost:8000] [docs/assets]
// Needs the server running and scripts/pdf/node_modules (playwright-core); uses the installed Chrome/Edge.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(here, "pdf", "package.json"));
const { chromium } = require("playwright-core");
const base = process.argv[2] || "http://localhost:8000";
const out = path.resolve(here, "..", process.argv[3] || "docs/assets");
fs.mkdirSync(out, { recursive: true });

const exe = [process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean).find((p) => fs.existsSync(p));
if (!exe) { console.error("No Chrome/Edge found (set CHROME_PATH)"); process.exit(2); }

const browser = await chromium.launch({ executablePath: exe, headless: true });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
const problems = [];
page.on("console", (m) => { if (m.type() === "error") problems.push("console: " + m.text()); });
page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
const wait = (ms) => page.waitForTimeout(ms);
const shot = async (name, opts = {}) => { await page.screenshot({ path: path.join(out, name), ...opts }); console.log("saved", name); };

await page.goto(base + "/", { waitUntil: "load" });
await wait(11000);                                      // let the five live streams fill the console
await shot("live_hud.png");               // the five codes side by side

// one code full screen (BB72)
await page.click('.card[data-c="BB72"]');
await wait(5000);

// weave + schedule: show one CNOT layer of the interleaved schedule
await page.evaluate(() => { const r = document.getElementById("lay"); r.value = 2; r.dispatchEvent(new Event("input")); });
await wait(1200);
await shot("tanner_explorer.png");

// sliding-window mode in live telemetry
await page.click('#modos button[data-m="win"]');
await wait(9000);
await shot("window_mode.png");

await page.evaluate(() => document.querySelector("#prueba").scrollIntoView());
await wait(900);
await shot("benchmarks.png");
await page.evaluate(() => document.querySelector("#circuito").scrollIntoView());
await wait(1200);
await shot("qasm_inspector.png");

await browser.close();
if (problems.length) { console.error("PAGE PROBLEMS:\n" + problems.join("\n")); process.exit(3); }
console.log("no console errors");
