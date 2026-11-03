"""Stitch the frames written by scripts/record_demo.mjs into an animated GIF (and WebP).

    python scripts/stitch_demo.py [frames-dir] [out-gif] [--width 1000]

Pillow only.  Frames are resized, quantized to a shared adaptive palette and the sequence loops.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="?", default=str(ROOT / "dist" / "demo_frames"))
    ap.add_argument("out", nargs="?", default=str(ROOT / "dist" / "outreach_ready" / "tars_live_demo.gif"))
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--ms", type=int, default=450, help="milliseconds per frame")
    a = ap.parse_args()
    files = sorted(Path(a.frames).glob("*.png"))
    if not files:
        raise SystemExit("no frames found")
    frames = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im = im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS)
        frames.append(im)
    # one shared palette built from a spread of frames keeps colours stable between frames
    sample = Image.new("RGB", (a.width, frames[0].height * min(6, len(frames))))
    for i, fr in enumerate(frames[:: max(1, len(frames) // 6)][:6]):
        sample.paste(fr, (0, i * fr.height))
    pal = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=a.ms, loop=0, optimize=True, disposal=1)
    webp = out.with_suffix(".webp")
    frames[0].save(webp, save_all=True, append_images=frames[1:], duration=a.ms, loop=0, quality=70, method=4)
    print(f"{len(frames)} frames | {out.name}: {out.stat().st_size / 1e6:.1f} MB | {webp.name}: {webp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
