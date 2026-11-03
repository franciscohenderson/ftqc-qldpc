"""Build dist/outreach_ready/LAUNCHPAD.html from dist/outreach_ready/manifest.json and open it in Chrome.

    python scripts/stage_outreach.py [--no-open]

The launchpad only PREPARES things: each button opens a pre-filled Gmail draft / mailto link / social composer or
copies text to the clipboard.  Nothing is sent or posted by this script; you review and press Send / Post yourself.
Contacts come from the manifest (written next to the outreach files); only addresses verified on the companies' own
pages are used, the others open the official contact route with an empty "To" field.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "outreach_ready"


def gmail(to: str, subject: str, body: str, sender: str = "") -> str:
    """Gmail compose URL; `sender` (authuser) makes Gmail open the draft from that signed-in account."""
    acct = f"&authuser={quote(sender)}" if sender else ""
    return f"https://mail.google.com/mail/?view=cm&fs=1{acct}&to={quote(to)}&su={quote(subject)}&body={quote(body)}"


def mailto(to: str, subject: str, body: str) -> str:
    return f"mailto:{quote(to)}?subject={quote(subject)}&body={quote(body.replace(chr(10), chr(13) + chr(10)))}"


def parse_answers(md: str) -> dict[str, list[tuple[str, str]]]:
    """{'Y Combinator': [(question, answer), ...], 'Creative Destruction Lab, Quantum stream': [...]}"""
    out: dict[str, list[tuple[str, str]]] = {}
    sec, q, buf = None, None, []

    def flush():
        nonlocal q, buf
        if sec and q:
            out[sec].append((q, "\n".join(buf).strip()))
        q, buf = None, []

    for line in md.splitlines():
        if line.startswith("## "):
            flush()
            sec = line[3:].strip()
            out[sec] = []
        elif sec and re.fullmatch(r"\*\*[^*].*[^*]\*\*", line.strip()) and not line.strip().startswith("- "):
            flush()
            q = line.strip().strip("*")
        elif q is not None:
            buf.append(line)
    flush()
    return out


CSS = """
:root{--lino:#d9d2bf;--papel:#e6e0cf;--tinta:#16204a;--cochi:#a3162d;--verdin:#287a5e;--mut:#5b5a52}
*{box-sizing:border-box}body{margin:0;background:var(--lino);color:var(--tinta);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 20px 80px}h1{font-size:2rem;margin:0 0 6px;letter-spacing:-.02em}h2{margin:34px 0 10px;border-bottom:1.5px solid var(--tinta);padding-bottom:6px}
.warn{background:var(--papel);border-left:4px solid var(--cochi);padding:10px 14px;margin:14px 0}
.card{background:var(--papel);border:1px solid rgba(22,32,74,.35);padding:14px 16px;margin:12px 0}
.card h3{margin:0 0 4px}.meta{color:var(--mut);font-size:13px;margin:2px 0}.to b{color:var(--tinta)}.missing{color:var(--cochi);font-weight:600}
pre{white-space:pre-wrap;background:#f3efe3;border:1px solid rgba(22,32,74,.2);padding:10px;max-height:190px;overflow:auto;font:13px/1.45 ui-monospace,Consolas,monospace}
.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
a.btn,button{font:600 13px "Segoe UI",sans-serif;border:1.5px solid var(--tinta);background:var(--tinta);color:#fff;padding:7px 12px;text-decoration:none;cursor:pointer}
a.btn.alt,button.alt{background:transparent;color:var(--tinta)}button.ok{background:var(--verdin);border-color:var(--verdin)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
"""

JS = """
function copyText(btn, text){
  const done=()=>{const o=btn.textContent;btn.classList.add('ok');btn.textContent='Copied';setTimeout(()=>{btn.classList.remove('ok');btn.textContent=o},1400)};
  if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(text).then(done,fallback)}else fallback();
  function fallback(){const t=document.createElement('textarea');t.value=text;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();try{document.execCommand('copy');done()}catch(e){alert('Copy failed: select the text manually')}t.remove()}
}
document.addEventListener('click',e=>{const b=e.target.closest('[data-copy]');if(b){copyText(b,DATA[b.dataset.copy])}});
"""


def build(manifest: dict, answers: dict, files: dict[str, bool]) -> str:
    data: dict[str, str] = {}
    parts: list[str] = []
    esc = html.escape

    # ---------------- emails
    parts.append("<h2>1. Emails (4 messages, 5 recipients)</h2>")
    for i, e in enumerate(manifest["emails"]):
        k = f"e{i}"
        data[k + "s"], data[k + "b"] = e["subject"], e["body"]
        to = e["to"]
        to_html = f'<span class="to"><b>{esc(to)}</b></span>' if to else '<span class="missing">no verified public address: use the official route below</span>'
        row = (f'<a class="btn" target="_blank" rel="noopener" href="{esc(gmail(to, e["subject"], e["body"], manifest.get("sender", "")))}">Open Gmail draft</a>'
               f'<a class="btn alt" href="{esc(mailto(to, e["subject"], e["body"]))}">mailto</a>'
               f'<button class="alt" data-copy="{k}s">Copy subject</button><button class="alt" data-copy="{k}b">Copy body</button>'
               f'<a class="btn alt" target="_blank" rel="noopener" href="{esc(e["route"])}">Official contact route</a>')
        parts.append(f'<div class="card"><h3>{esc(e["company"])} <small style="font-weight:400;color:var(--mut)">({esc(e["label"])})</small></h3>'
                     f'<div class="meta">To: {to_html}</div><div class="meta">{esc(e["note"])}</div>'
                     f'<div class="meta">Subject: <b>{esc(e["subject"])}</b></div><pre>{esc(e["body"])}</pre><div class="row">{row}</div></div>')

    # ---------------- social
    parts.append("<h2>2. LinkedIn and X posts</h2>")
    parts.append('<div class="warn">LinkedIn does not reliably pre-fill text from a link, so the composer opens and the post is also copied with one click: paste it, then drag in the GIF and the PDF (the folder is open in Explorer).</div>')
    parts.append('<div class="grid">')
    for lang, label in (("en", "English"), ("es", "Español")):
        k = f"p{lang}"
        data[k] = manifest["posts"][lang]
        li = "https://www.linkedin.com/feed/?shareActive=true&text=" + quote(manifest["posts"][lang])
        parts.append(f'<div class="card"><h3>LinkedIn ({label})</h3><pre>{esc(manifest["posts"][lang])}</pre><div class="row">'
                     f'<button data-copy="{k}">Copy post</button><a class="btn alt" target="_blank" rel="noopener" href="{esc(li)}">Open LinkedIn composer</a></div></div>')
    for lang, label in (("en", "English"), ("es", "Español")):
        k = f"x{lang}"
        txt = manifest["x"][lang]
        data[k] = txt
        intent = "https://twitter.com/intent/tweet?text=" + quote(txt)
        parts.append(f'<div class="card"><h3>X / Twitter ({label})</h3><pre>{esc(txt)}</pre><div class="meta">{len(txt) - len(manifest["release_url"]) + 23} characters counting the link as 23</div><div class="row">'
                     f'<button data-copy="{k}">Copy</button><a class="btn alt" target="_blank" rel="noopener" href="{esc(intent)}">Open X compose</a></div></div>')
    parts.append("</div>")
    media = []
    if files.get("gif"):
        media.append('<a class="btn alt" href="tars_live_demo.gif" target="_blank">Preview GIF</a>')
    media.append('<a class="btn alt" href="FTQC_qLDPC_Whitepaper_2026.pdf" target="_blank">Open PDF</a>')
    media.append(f'<button class="alt" data-copy="rel">Copy PDF download link</button>')
    data["rel"] = manifest["pdf_url"]
    parts.append(f'<div class="card"><h3>Media to attach</h3><div class="meta">Hosted PDF: {esc(manifest["pdf_url"])}</div><div class="row">{"".join(media)}</div></div>')

    # ---------------- accelerators
    parts.append("<h2>3. Accelerator answers (copy per question or the whole set)</h2>")
    n = 0
    for sec, qa in answers.items():
        allk = f"all{n}"
        data[allk] = "\n\n".join(f"{q}\n{a}" for q, a in qa)
        parts.append(f'<div class="card"><h3>{esc(sec)}</h3><div class="row"><button data-copy="{allk}">Copy all answers ({len(qa)})</button></div></div>')
        for q, a in qa:
            k = f"a{n}"
            n += 1
            data[k] = a
            parts.append(f'<div class="card"><div class="meta"><b>{esc(q)}</b></div><pre>{esc(a)}</pre><div class="row"><button class="alt" data-copy="{k}">Copy answer</button></div></div>')
        n += 1
    body = "\n".join(parts)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Launchpad: ftqc-qldpc outreach</title><style>{CSS}</style></head><body><main>"
            f'<h1>Launchpad</h1><div class="meta">{esc(manifest["repo"])} · release v0.1.0</div>'
            f'<div class="warn"><b>Nothing is sent from this page.</b> Each button opens a pre-filled draft or copies text; you read it, edit it if you want, and press Send or Post yourself.</div>'
            f"{body}</main><script>const DATA={payload};{JS}</script></body></html>")


def chrome() -> str | None:
    for p in (os.environ.get("CHROME_PATH"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", shutil.which("google-chrome"), shutil.which("chromium")):
        if p and Path(p).exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    man = OUT / "manifest.json"
    if not man.exists():
        sys.exit(f"{man} not found: generate the outreach bundle first")
    manifest = json.loads(man.read_text(encoding="utf-8"))
    answers = parse_answers((OUT / "05_YC_CDL_Accelerator_Application_Answers.md").read_text(encoding="utf-8"))
    page = build(manifest, answers, {"gif": (OUT / "tars_live_demo.gif").exists()})
    dest = OUT / "LAUNCHPAD.html"
    dest.write_text(page, encoding="utf-8")
    print(f"wrote {dest} ({len(page) // 1024} KB; {sum(len(v) for v in answers.values())} answers, {len(manifest['emails'])} emails)")
    if a.no_open:
        return
    exe = chrome()
    if exe:
        subprocess.Popen([exe, "--new-window", dest.as_uri()])
    elif sys.platform == "win32":
        os.startfile(dest)  # noqa: S606 (default browser)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", str(OUT)])


if __name__ == "__main__":
    main()
