// Record frames of the live console for an animated demo.  Usage:
//   node scripts/record_demo.mjs [http://localhost:8000] [frames-dir]
// Then stitch with: python scripts/stitch_demo.py <frames-dir>
// Needs the server running and scripts/pdf/node_modules (playwright-core); uses the installed Chrome/Edge.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { chromium } = createRequire(path.join(here, "pdf", "package.json"))("playwright-core");
const base = process.argv[2] || "http://localhost:8000";
const dir = path.resolve(process.argv[3] || path.join(here, "..", "dist", "demo_frames"));
fs.rmSync(dir, { recursive: true, force: true });
fs.mkdirSync(dir, { recursive: true });

const exe = [process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean).find((p) => fs.existsSync(p));
if (!exe) { console.error("No Chrome/Edge found (set CHROME_PATH)"); process.exit(2); }

const browser = await chromium.launch({ executablePath: exe, headless: true });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
const problems = [];
page.on("pageerror", (e) => problems.push(e.message));
let n = 0;
const wait = (ms) => page.waitForTimeout(ms);
const snap = async (caption) => { const f = path.join(dir, String(n++).padStart(3, "0") + ".png"); await page.screenshot({ path: f }); fs.writeFileSync(f + ".txt", caption); };
const burst = async (count, gap, caption) => { for (let i = 0; i < count; i++) { await snap(caption); await wait(gap); } };

await page.goto(base + "/", { waitUntil: "load" });
await wait(7000);                                          // live stream fills the console (code-capacity mode)
await burst(8, 350, "Live decoding: syndromes light the threads, the corrections close them");

// the 7 layers of the interleaved CNOT schedule
for (let layer = 0; layer < 7; layer++) {
  await page.evaluate((l) => { const r = document.getElementById("lay"); r.value = l; r.dispatchEvent(new Event("input")); }, layer);
  await wait(250);
  await burst(2, 220, `Depth-7 interleaved schedule: layer ${layer + 1} of 7`);
}
await page.click("#tS");                                   // sequential schedule for contrast (12 layers)
await wait(300);
await burst(2, 250, "Sequential schedule: 12 layers");
await page.click("#tI");

await page.click('#modes button[data-m="full"]');          // full-horizon circuit decoding
await wait(3500);
await burst(5, 600, "Full-horizon circuit decoding");

await page.click('#modes button[data-m="win"]');           // sliding-window (W,F)
await wait(3500);
await burst(7, 600, "Sliding-window (W,F) decoding: much shorter wait after the last round");

await browser.close();
if (problems.length) { console.error("PAGE ERRORS:\n" + problems.join("\n")); process.exit(3); }
console.log("frames:", n, "->", dir);
