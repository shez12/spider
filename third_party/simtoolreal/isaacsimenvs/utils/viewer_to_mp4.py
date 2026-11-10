"""Record a pose-viewer HTML replay into an mp4 (headless Chrome + CDP).

Isaac Sim's viewport recording needs the RTX renderer, which segfaults on some
headless machines (``create_new_stage`` → hydra engine).  This tool renders the
already-generated three.js viewer page in headless Chrome (SwiftShader) and
screenshots it frame by frame, then encodes the frames with the ffmpeg binary
bundled with imageio — no system ffmpeg, no GPU, no Isaac cameras.

    python isaacsimenvs/utils/viewer_to_mp4.py outputs/traj_eval/x/ep0_viewer.html \
        -o ep0.mp4 --fps 30 --stride 2

Control is 60 Hz, so ``--stride 2 --fps 30`` gives real-time playback.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = (
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def find_chrome(explicit: str | None = None) -> str:
    """Locate a Chrome/Chromium binary."""
    if explicit:
        return explicit
    for candidate in CHROME_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("google-chrome") or shutil.which("chromium")
    if found:
        return found
    raise RuntimeError("no Chrome/Chromium binary found; pass --chrome /path/to/chrome")


def _draw_overlay(frame, text: str):
    """Stamp a caption onto an RGB frame (top-left, on a dark plate)."""
    import numpy as _np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    pad = 8
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.rectangle([10, 10, 10 + w + 2 * pad, 10 + h + 2 * pad], fill=(0, 0, 0))
    draw.text((10 + pad, 10 + pad), text, fill=(255, 255, 255), font=font)
    return _np.asarray(img)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def record_viewer_mp4(
    html_path: str | Path,
    out_path: str | Path,
    fps: int = 30,
    stride: int = 2,
    width: int = 1280,
    height: int = 720,
    max_frames: int | None = None,
    chrome: str | None = None,
    startup_timeout_s: float = 120.0,
    verbose: bool = True,
    overlay: str | None = None,
) -> Path:
    """Screenshot the viewer page frame by frame and encode it to mp4.

    Returns the mp4 path.  Raises RuntimeError if the page never becomes ready.
    """
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    from websockets.sync.client import connect

    html_path = Path(html_path).resolve()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chrome_bin = find_chrome(chrome)
    port = _free_port()
    import tempfile

    profile_dir = tempfile.mkdtemp(prefix="viewer2mp4_chrome_")

    proc = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu-sandbox",
            "--allow-file-access-from-files",  # URDF meshes live next to the HTML
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={width},{height}",
            "--no-first-run",
            "--disable-extensions",
            f"--user-data-dir={profile_dir}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def log(msg: str) -> None:
        if verbose:
            print(f"[viewer2mp4] {msg}", flush=True)

    try:
        # --- connect ---
        deadline = time.time() + startup_timeout_s
        ws_url = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                    targets = json.load(r)
                pages = [t for t in targets if t["type"] == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.4)
        if ws_url is None:
            raise RuntimeError("could not reach the Chrome DevTools endpoint")

        with connect(ws_url, max_size=None, open_timeout=30) as ws:
            msg_id = 0

            def send(method: str, params: dict | None = None) -> dict:
                nonlocal msg_id
                msg_id += 1
                ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == msg_id:
                        if "error" in msg:
                            raise RuntimeError(f"CDP {method}: {msg['error']}")
                        return msg

            def evaluate(expression: str, await_promise: bool = False):
                res = send(
                    "Runtime.evaluate",
                    {
                        "expression": expression,
                        "awaitPromise": await_promise,
                        "returnByValue": True,
                    },
                )
                result = res["result"]["result"]
                return result.get("value")

            send("Page.enable")
            send("Runtime.enable")
            send("Emulation.setDeviceMetricsOverride", {
                "width": width,
                "height": height,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            send("Page.navigate", {"url": f"file://{html_path}"})

            # --- wait until the URDF robot finished loading ---
            # applyFrame() is a no-op until `animatedRobot` exists, and the
            # module scope is not reachable from CDP — drive it through the
            # timeline slider (its "input" handler calls applyFrame) and watch
            # the frame readout for confirmation.
            seek_js = """
            (() => {
              const tl = document.getElementById('timeline');
              const meta = document.getElementById('frame-meta');
              if (!tl) return null;
              tl.value = String(%d);
              tl.dispatchEvent(new Event('input', {bubbles: true}));
              const err = document.getElementById('error');
              return {
                max: Number(tl.max),
                meta: meta ? meta.textContent : '',
                error: (err && err.style.display !== 'none') ? err.textContent : ''
              };
            })()
            """
            info = None
            while time.time() < deadline:
                try:
                    info = evaluate(seek_js % 0)
                except Exception:
                    info = None
                if info and info.get("error"):
                    raise RuntimeError(f"viewer page reported an error: {info['error']}")
                if info and info.get("max", 0) > 0 and "Frame" in (info.get("meta") or ""):
                    break
                time.sleep(0.5)
            if not info or info.get("max", 0) <= 0:
                raise RuntimeError("viewer page never became ready (timeline not initialized)")

            num_frames = int(info["max"]) + 1
            if max_frames:
                num_frames = min(num_frames, int(max_frames))
            log(f"{num_frames} frames available, recording every {max(1, stride)} -> {out_path.name}")

            # Hide the UI chrome so the mp4 is just the 3D viewport.
            evaluate(
                """
                (() => {
                  const ids = ['controls', 'titlebar'];
                  ids.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.style.display = 'none';
                  });
                  window.dispatchEvent(new Event('resize'));
                  return true;
                })()
                """
            )
            time.sleep(0.5)

            writer = imageio.get_writer(
                str(out_path), fps=int(fps), codec="libx264", quality=8, macro_block_size=1
            )
            drawn = 0
            try:
                for i in range(0, num_frames, max(1, stride)):
                    evaluate(seek_js % i)
                    # let three.js render the new pose
                    evaluate(
                        "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))",
                        await_promise=True,
                    )
                    shot = send("Page.captureScreenshot", {"format": "jpeg", "quality": 92})
                    raw = base64.b64decode(shot["result"]["data"])
                    frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                    if overlay:
                        frame = _draw_overlay(frame, overlay)
                    writer.append_data(frame)
                    drawn += 1
                    if verbose and drawn % 50 == 0:
                        log(f"{drawn} frames captured ({i}/{num_frames})")
            finally:
                writer.close()

        log(f"wrote {drawn} frames -> {out_path}")
        return out_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        shutil.rmtree(profile_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", help="Pose-viewer HTML produced by SimToolRealPoseViewerWrapper")
    parser.add_argument("-o", "--output", default=None, help="Output mp4 (default: <html stem>.mp4)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2, help="Capture every N-th frame")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--chrome", default=None)
    parser.add_argument("--overlay", default=None, help="Caption stamped onto every frame")
    args = parser.parse_args()

    out = args.output or str(Path(args.html).with_suffix(".mp4"))
    record_viewer_mp4(
        args.html,
        out,
        fps=args.fps,
        stride=args.stride,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        chrome=args.chrome,
        overlay=args.overlay,
    )
    print(f"[viewer2mp4] done: {out}")


if __name__ == "__main__":
    main()
