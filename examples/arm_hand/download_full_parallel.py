"""Resume the release archive via HTTP ranges, then run standard verification."""

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Preserve the downloaded prefix and verify a complete parallel transfer."""
    stage = ROOT / "example_datasets/.retarget_full-download"
    distribution = json.loads((stage / "transfer/distribution.json").read_text())
    revision = json.loads((stage / "candidate.json").read_text())["data_revision"]
    target = stage / "transfer" / distribution["archive"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert (
            hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
            == distribution["sha256"]
        )
        print("Complete archive already verified", flush=True)
        return
    state_path = stage / "parallel_download_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        partial = Path(state["partial"])
    else:
        partials = list(
            (stage / "transfer/.cache/huggingface/download/distribution").glob(
                "*.incomplete"
            )
        )
        partial = (
            max(partials, key=lambda p: p.stat().st_size)
            if partials
            else stage / "parallel_archive.part"
        )
        partial.touch(exist_ok=True)
        state = {
            "partial": str(partial),
            "prefix_bytes": partial.stat().st_size,
            "completed": [],
        }
    prefix = state["prefix_bytes"]
    size = distribution["bytes"]
    chunk_size = 64 * 1024 * 1024
    ranges = [
        (start, min(size - 1, start + chunk_size - 1))
        for start in range(prefix, size, chunk_size)
    ]
    done = set(state["completed"])
    lock = threading.Lock()

    def save():
        state["completed"] = sorted(done)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state) + "\n")
        temporary.replace(state_path)

    save()
    url = f"https://huggingface.co/datasets/retarget/retarget_full/resolve/{revision}/{distribution['archive']}"
    descriptor = os.open(partial, os.O_RDWR)

    def fetch(index):
        if index in done:
            return
        start, end = ranges[index]
        for attempt in range(10):
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(30, 600),
                ) as response:
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{size}"
                    ):
                        raise ValueError(
                            f"Unexpected range response: {response.status_code}, {response.headers.get('Content-Range')}"
                        )
                    position = start
                    for data in response.iter_content(2 * 1024 * 1024):
                        if position + len(data) > end + 1:
                            raise ValueError("Range response longer than requested")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(descriptor, view, position)
                            view = view[written:]
                            position += written
                    if position != end + 1:
                        raise ValueError("Incomplete HTTP range")
                with lock:
                    done.add(index)
                    save()
                    if len(done) % 8 == 0 or len(done) == len(ranges):
                        received = prefix + sum(
                            ranges[i][1] - ranges[i][0] + 1 for i in done
                        )
                        print(
                            f"Archive received: {received / 1e9:.2f}/{size / 1e9:.2f} GB",
                            flush=True,
                        )
                return
            except (requests.RequestException, ValueError):
                if attempt == 9:
                    raise RuntimeError(
                        f"HTTP range {index} failed after ten attempts"
                    ) from None

    try:
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(fetch, range(len(ranges))))
    finally:
        os.close(descriptor)
    with partial.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest() == distribution["sha256"]
        ), "Archive checksum mismatch"
    partial.replace(target)
    print("Full archive SHA256 verified:", distribution["sha256"], flush=True)


if __name__ == "__main__":
    main()
