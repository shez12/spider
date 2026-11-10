"""Compose several clips into one side-by-side comparison video.

Useful when the whole point is a *difference* between runs (e.g. the same
trajectory driven at several commanded speeds): playing them in a grid at the
same time makes the difference obvious, while watching five clips one after
another does not.

    python isaacsimenvs/utils/compose_video_grid.py \
        --clip outputs/speedcmd/videos/cmd0.05.mp4="command 0.05" \
        --clip outputs/speedcmd/videos/cmd0.35.mp4="command 0.35" \
        -o outputs/speedcmd/videos/compare.mp4 --cols 3 --fps 30

Clips are aligned by frame index (so all inputs should share one fps).  A clip
that ends early holds its last frame and is marked "done".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 20):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _stamp(img: Image.Image, lines: list[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _font()
    pad = 6
    widths, height = [], 0
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        height += box[3] - box[1] + 4
    w = max(widths)
    draw.rectangle([8, 8, 8 + w + 2 * pad, 8 + height + 2 * pad], fill=(0, 0, 0))
    y = 8 + pad
    for line in lines:
        draw.text((8 + pad, y), line, fill=(255, 255, 255), font=font)
        y += draw.textbbox((0, 0), line, font=font)[3] + 4
    return img


def compose(clips: list[tuple[str, str]], out_path: Path, cols: int, fps: int, cell=(640, 360)) -> None:
    import imageio.v2 as imageio

    paths = [c[0] for c in clips]
    labels = [c[1] for c in clips]
    print(f"[grid] opening {len(paths)} clips...")
    # Stream frame by frame: mimread() would buffer every clip in RAM.
    readers = [imageio.get_reader(p) for p in paths]
    streams = [iter(r) for r in readers]
    lengths = [int(r.count_frames()) for r in readers]
    total = max(lengths)
    print(f"[grid] lengths (frames): {lengths} -> {total} frames @ {fps} fps")

    rows = (len(paths) + cols - 1) // cols
    cw, ch = cell
    canvas_size = (cols * cw, rows * ch)

    held = [None] * len(readers)
    writer = imageio.get_writer(
        str(out_path), fps=int(fps), codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for i in range(total):
            canvas = Image.new("RGB", canvas_size, (16, 16, 16))
            for k in range(len(readers)):
                if i < lengths[k]:
                    held[k] = next(streams[k])
                frame = held[k]
                if frame is None:
                    continue
                tile = Image.fromarray(frame).resize((cw, ch))
                stamp = [labels[k]] + (["done"] if i >= lengths[k] else [])
                tile = _stamp(tile, stamp)
                r, c = divmod(k, cols)
                canvas.paste(tile, (c * cw, r * ch))
            writer.append_data(np.asarray(canvas))
            if i % 50 == 0:
                print(f"[grid] {i}/{total}", flush=True)
    finally:
        writer.close()
        for r in readers:
            try:
                r.close()
            except Exception:
                pass
    print(f"[grid] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip", action="append", required=True, help="path=label (repeatable)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    clips = []
    for spec in args.clip:
        path, _, label = spec.partition("=")
        clips.append((path, label or Path(path).stem))
    compose(clips, Path(args.output), cols=int(args.cols), fps=int(args.fps))


if __name__ == "__main__":
    main()
