"""Build docs/WHITEPAPER.pdf: two-column academic layout from docs/WHITEPAPER.md.

Pipeline: Markdown (python-markdown) -> HTML, LaTeX math rendered with KaTeX (node), vector SVG figures from
figures/, print stylesheet scripts/pdf/whitepaper.css, printed to PDF by headless Chrome/Chromium/Edge
(via playwright-core, no browser download).  Re-runnable; verifies the result.

    python scripts/build_pdf.py [--md docs/WHITEPAPER.md] [--out docs/WHITEPAPER.pdf]

Requirements: Python `markdown` and `pypdf`, Node >= 18, an installed Chrome/Chromium/Edge (CHROME_PATH to override).
`npm install` is run automatically in scripts/pdf on first use.
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

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "scripts" / "pdf"


def run_node(args: list[str], stdin: str | None = None) -> str:
    node = shutil.which("node")
    if not node:
        sys.exit("Node.js is required (https://nodejs.org).")
    r = subprocess.run([node, str(PDF_DIR / "render.mjs"), *args], input=stdin, capture_output=True, text=True, encoding="utf-8", cwd=PDF_DIR)
    if r.returncode:
        sys.exit(f"render.mjs {args[0]} failed:\n{r.stderr[-1500:]}")
    return r.stdout


def ensure_node_modules() -> None:
    if (PDF_DIR / "node_modules" / "katex").exists() and (PDF_DIR / "node_modules" / "playwright-core").exists():
        return
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        sys.exit("npm is required to install katex and playwright-core.")
    env = dict(os.environ, PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="1")
    subprocess.run([npm, "install", "--no-audit", "--no-fund", "--loglevel=error"], cwd=PDF_DIR, env=env, check=True)


def protect(pattern: re.Pattern, text: str, store: list[str], tag: str) -> str:
    def sub(m):
        store.append(m.group(0))
        return f"@@{tag}{len(store) - 1}@@"
    return pattern.sub(sub, text)


FENCE = re.compile(r"^```.*?^```[ \t]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]+`")
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$])\$(?!\s)([^\n$]+?)(?<![\s\\])\$")
IMAGE = re.compile(r"^!\[(.*)\]\((.+?)\)[ 	]*$", re.M)


LIST_ITEM = re.compile(r"^(\s*)([*+-]|\d+\.)\s+")


def fix_lists(text: str) -> str:
    """python-markdown needs a blank line before a list and 4-space nesting; GitHub does not."""
    out: list[str] = []
    for line in text.split("\n"):
        m = LIST_ITEM.match(line)
        if m:
            indent = len(m.group(1))
            if 0 < indent < 4:
                line = "    " + line.lstrip()
            prev = out[-1] if out else ""
            if prev.strip() and not LIST_ITEM.match(prev) and not prev.startswith((" ", "\t", "|")):
                out.append("")
        out.append(line)
    return "\n".join(out)


def convert(md_text: str, md_dir: Path) -> tuple[str, list[dict]]:
    import markdown

    code_blocks: list[str] = []
    text = protect(FENCE, md_text, code_blocks, "CB")
    inline_code: list[str] = []
    text = protect(INLINE_CODE, text, inline_code, "IC")
    maths: list[dict] = []

    def disp(m):
        maths.append({"tex": m.group(1).strip(), "display": True})
        return f"@@M{len(maths) - 1}@@"

    def inl(m):
        maths.append({"tex": m.group(1), "display": False})
        return f"@@M{len(maths) - 1}@@"

    text = DISPLAY.sub(disp, text)
    text = INLINE.sub(inl, text)
    figs: list[tuple[str, str]] = []

    def fig(m):
        figs.append((m.group(1), m.group(2)))
        return f"@@F{len(figs) - 1}@@"

    text = IMAGE.sub(fig, text)
    text = fix_lists(text)
    for i, c in enumerate(inline_code):
        text = text.replace(f"@@IC{i}@@", c)
    for i, c in enumerate(code_blocks):
        text = text.replace(f"@@CB{i}@@", c)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5")
    # figures
    for i, (cap, src) in enumerate(figs):
        path = (md_dir / src).resolve()
        wide = cap.startswith("[wide]")
        cap = cap[6:].strip() if wide else cap
        label = re.match(r"(Figure \d+\.)\s*(.*)", cap, re.S)
        cap_html = f"<b>{label.group(1)}</b> {label.group(2)}" if label else cap
        body = body.replace(f"<p>@@F{i}@@</p>", f'<figure class="{"wide" if wide else ""}"><img src="{path.as_uri()}" alt="figure"><figcaption>{cap_html}</figcaption></figure>')
    return body, maths


def restore_math(body: str, maths: list[dict]) -> str:
    if not maths:
        return body
    rendered = json.loads(run_node(["math"], json.dumps(maths)))
    for i in range(len(maths) - 1, -1, -1):
        body = body.replace(f"@@M{i}@@", rendered[i])
    return body


def postprocess(body: str) -> str:
    # table captions written as headings -> caption paragraphs
    body = re.sub(r"<h[34]>(Table \d+.*?)</h[34]>", r'<p class="tcap">\1</p>', body, flags=re.S)
    # wide tables span both columns together with their caption
    def widen(m):
        table = m.group(0)
        n_cols = len(re.findall(r"<th", table.split("</tr>")[0]))
        return table.replace("<table>", '<table class="wide">', 1) if n_cols >= 6 else table
    body = re.sub(r"<table>.*?</table>", widen, body, flags=re.S)
    body = re.sub(r'(<p class="tcap">.*?</p>)\s*(<table class="wide">.*?</table>)', r'<div class="spanall">\1\2</div>', body, flags=re.S)
    return body


def build(md_path: Path, out_pdf: Path) -> dict:
    ensure_node_modules()
    raw = md_path.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", raw, re.M).group(1).strip()
    authors = (re.search(r"^\*\*Authors:\*\*\s*(.+)$", raw, re.M) or [None, "[to be filled]"])[1] if re.search(r"^\*\*Authors:\*\*", raw, re.M) else ""
    abs_m = re.search(r"^## Abstract\s*\n(.*?)(?=^---\s*$|^## )", raw, re.S | re.M)
    abstract = abs_m.group(1).strip() if abs_m else ""
    rest = raw[abs_m.end():] if abs_m else raw
    # split references into their own block
    parts = re.split(r"^## References\s*$", rest, maxsplit=1, flags=re.M)
    main_md, refs_md = parts[0], (parts[1] if len(parts) > 1 else "")
    pieces = {"abstract": abstract, "main": main_md, "refs": "## References\n" + refs_md if refs_md.strip() else ""}
    htmls, all_maths = {}, []
    # one math pass over all pieces (indices are global)
    offset = 0
    for k, md in pieces.items():
        if not md.strip():
            htmls[k] = ""
            continue
        body, maths = convert(md, md_path.parent)
        body = re.sub(r"@@M(\d+)@@", lambda m: f"@@M{int(m.group(1)) + offset}@@", body)
        offset += len(maths)
        all_maths += maths
        htmls[k] = body
    title_html, authors_html = html.escape(title), html.escape(authors)
    # the title is plain text; math placeholders in headings are restored below
    full = "".join(htmls.values())
    rendered = json.loads(run_node(["math"], json.dumps(all_maths))) if all_maths else []
    for k in htmls:
        for i in range(len(all_maths) - 1, -1, -1):
            htmls[k] = htmls[k].replace(f"@@M{i}@@", rendered[i])
        htmls[k] = postprocess(htmls[k])
    errs = sum(h.count("katex-error") for h in htmls.values())
    build_dir = ROOT / "docs" / ".pdf_build"
    shutil.rmtree(build_dir, ignore_errors=True)
    (build_dir / "fonts").mkdir(parents=True)
    kdist = PDF_DIR / "node_modules" / "katex" / "dist"
    shutil.copy(kdist / "katex.min.css", build_dir / "katex.min.css")
    for f in (kdist / "fonts").glob("*.woff2"):
        shutil.copy(f, build_dir / "fonts" / f.name)
    css = (PDF_DIR / "whitepaper.css").read_text(encoding="utf-8")
    abstract_html = htmls["abstract"].replace("<p>", '<p><b class="lead">Abstract.</b> ', 1) if htmls["abstract"] else ""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title_html}</title>
<link rel="stylesheet" href="katex.min.css"><style>{css}</style><style>.spanall{{column-span:all}}</style></head><body>
<header class="title"><h1>{title_html}</h1><div class="authors">{authors_html}</div><div class="abstract">{abstract_html}</div></header>
<main>{htmls["main"]}<div class="refs">{htmls["refs"]}</div></main></body></html>"""
    html_path = build_dir / "whitepaper.html"
    html_path.write_text(doc, encoding="utf-8")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    run_node(["pdf", str(html_path), str(out_pdf), title])
    return {"title": title, "katex_errors": errs, "math_items": len(all_maths), "html": str(html_path)}


def verify(pdf: Path, info: dict) -> None:
    from pypdf import PdfReader

    assert pdf.exists() and pdf.stat().st_size > 20_000, "PDF missing or suspiciously small"
    r = PdfReader(str(pdf))
    n = len(r.pages)
    texts = [(p.extract_text() or "") for p in r.pages]
    problems = []
    if info["katex_errors"]:
        problems.append(f"{info['katex_errors']} KaTeX parse errors")
    if "@@" in "".join(texts):
        problems.append("unreplaced placeholder tokens (@@) in the PDF text")
    if info["title"][:30] not in texts[0].replace("\n", " "):
        problems.append("title not found on page 1")
    for i, t in enumerate(texts, 1):
        if len(t.strip()) < 250:
            problems.append(f"page {i} is nearly empty ({len(t.strip())} chars)")
    if "references" not in "".join(texts[-3:]).lower():
        problems.append("References not found in the last pages")
    w, h = float(r.pages[0].mediabox.width), float(r.pages[0].mediabox.height)
    print(f"PDF ok: {pdf} | {n} pages | {pdf.stat().st_size / 1024:.0f} KB | {w:.0f}x{h:.0f} pt | {info['math_items']} equations | KaTeX errors: {info['katex_errors']}")
    if problems:
        sys.exit("VERIFICATION FAILED:\n - " + "\n - ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "docs" / "WHITEPAPER.md"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "WHITEPAPER.pdf"))
    a = ap.parse_args()
    info = build(Path(a.md), Path(a.out))
    verify(Path(a.out), info)


if __name__ == "__main__":
    main()
