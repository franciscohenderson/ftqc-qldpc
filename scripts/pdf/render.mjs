// Helper for scripts/build_pdf.py
//   node render.mjs math  < jobs.json   > out.json     jobs: [{tex, display}] -> [html]
//   node render.mjs pdf   in.html out.pdf [headerTitle]
import fs from "node:fs";
import katex from "katex";

const [, , mode, a, b, c] = process.argv;

if (mode === "math") {
  const jobs = JSON.parse(fs.readFileSync(0, "utf8"));
  const out = jobs.map(({ tex, display }) =>
    katex.renderToString(tex, { displayMode: !!display, throwOnError: false, output: "html", strict: "ignore", trust: false, macros: {} }));
  process.stdout.write(JSON.stringify(out));
} else if (mode === "pdf") {
  const { chromium } = await import("playwright-core");
  const candidates = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ].filter(Boolean);
  const exe = candidates.find((p) => fs.existsSync(p));
  if (!exe) { console.error("No Chrome/Chromium/Edge found. Set CHROME_PATH."); process.exit(2); }
  const browser = await chromium.launch({ executablePath: exe, headless: true });
  const page = await browser.newPage();
  await page.goto("file:///" + a.replace(/\\/g, "/"), { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(600);
  const title = (c || "").replace(/[&<>]/g, "");
  await page.pdf({
    path: b, format: "Letter", printBackground: true, preferCSSPageSize: false,
    margin: { top: "0.75in", bottom: "0.8in", left: "0.65in", right: "0.65in" },
    displayHeaderFooter: true,
    headerTemplate: `<div style="font:7.5pt 'Times New Roman',serif;color:#555;width:100%;text-align:center;padding-top:6px">${title}</div>`,
    footerTemplate: `<div style="font:8pt 'Times New Roman',serif;color:#333;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
  });
  await browser.close();
} else {
  console.error("usage: render.mjs math|pdf ...");
  process.exit(1);
}
