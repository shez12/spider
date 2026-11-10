"""Download the official full DexYCB release with resumable verified byte ranges."""

import concurrent.futures
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "example_datasets/raw/dexycb"
sys.path.insert(0, str(DEST / ".download-tools"))
import requests  # noqa: E402
from gdown.download import get_url_from_gdrive_confirmation  # noqa: E402

FILE_ID = "18fD8RtWJM_fBi3bsuQTzdjIKxdRYtl57"
TOTAL = 127611213470
CHUNK = 128 * 1024 * 1024
DOWNLOADS = DEST / "downloads"
PARTS = DOWNLOADS / "parts"
STATUS = DEST / "download_status.json"
lock = threading.Lock()
progress = {}


def status(phase, **fields):
    """Atomically persist progress for inspection after disconnects."""
    data = dict(
        phase=phase,
        updated_unix=time.time(),
        total_bytes=TOTAL,
        source="https://dex-ycb.github.io/",
        file_id=FILE_ID,
        **fields,
    )
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(STATUS)
    print(json.dumps(data), flush=True)


def resolve():
    """Follow the public Google Drive confirmation form."""
    session = requests.Session()
    r = session.get(f"https://drive.google.com/uc?id={FILE_ID}", timeout=60)
    r.raise_for_status()
    url = get_url_from_gdrive_confirmation(r.text)
    r.close()
    return session, url


def fetch(index):
    """Validate the exact response range and persist one independent chunk."""
    start = index * CHUNK
    end = min(TOTAL, start + CHUNK) - 1
    size = end - start + 1
    path = PARTS / f"{index:05d}.part"
    if path.exists() and path.stat().st_size == size:
        with lock:
            progress[index] = size
        return
    tmp = path.with_suffix(".tmp")
    for attempt in range(10):
        try:
            session, url = resolve()
            with (
                session,
                session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(30, 600),
                ) as response,
            ):
                response.raise_for_status()
                expected = f"bytes {start}-{end}/{TOTAL}"
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != expected
                ):
                    raise RuntimeError("Server did not return the requested byte range")
                received = 0
                with tmp.open("wb") as f:
                    for block in response.iter_content(1024 * 1024):
                        f.write(block)
                        received += len(block)
                        if received > size:
                            raise RuntimeError("Oversized response")
                        with lock:
                            progress[index] = received
                if received != size:
                    raise RuntimeError("Truncated response")
                tmp.replace(path)
                return
        except Exception as error:
            with lock:
                progress[index] = 0
            print(
                f"Chunk {index} retry {attempt + 1}: {type(error).__name__}: {error}",
                flush=True,
            )
            if attempt == 9:
                raise
            time.sleep(min(30, 2**attempt))


def main():
    """Download, assemble, extract, and validate a real MANO annotation sample."""
    PARTS.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOADS / "dex-ycb-20210415.tar.gz"
    if not archive.exists():
        count = math.ceil(TOTAL / CHUNK)
        # Reuse complete chunks from the initial sequential transfer.
        for old in DOWNLOADS.glob("dex-ycb-20210415.tar.gz*.part"):
            with old.open("rb") as source:
                for i in range(old.stat().st_size // CHUNK):
                    path = PARTS / f"{i:05d}.part"
                    block = source.read(CHUNK)
                    if not path.exists():
                        path.write_bytes(block)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            pending = {pool.submit(fetch, i) for i in range(count)}
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=15)
                for future in done:
                    future.result()
                with lock:
                    downloaded = sum(progress.values())
                status(
                    "downloading",
                    downloaded_bytes=downloaded,
                    progress_percent=100 * downloaded / TOTAL,
                    pending_chunks=len(pending),
                )
                if not pending:
                    break
            # Verify every part before accepting or assembling the archive.
        for i in range(count):
            if (PARTS / f"{i:05d}.part").stat().st_size != min(
                CHUNK, TOTAL - i * CHUNK
            ):
                raise RuntimeError(f"Incomplete chunk {i}")
        status("assembling", downloaded_bytes=TOTAL)
        digest = hashlib.sha256()
        temp = archive.with_suffix(".assembling")
        with temp.open("wb") as output:
            for i in range(count):
                with (PARTS / f"{i:05d}.part").open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                        digest.update(block)
        assert temp.stat().st_size == TOTAL
        temp.replace(archive)
        (DEST / "archive_sha256.txt").write_text(
            digest.hexdigest() + "  " + archive.name + "\n"
        )
    status("extracting", downloaded_bytes=TOTAL)
    subprocess.run(
        [
            "tar",
            "--extract",
            "--gzip",
            "--file",
            str(archive),
            "--directory",
            str(DEST),
            "--no-same-owner",
            "--no-same-permissions",
        ],
        check=True,
    )
    status("validating", downloaded_bytes=TOTAL)
    import numpy as np

    labels = list(DEST.glob("2020*-subject-*/*/*/labels_*.npz"))
    if (
        not labels
        or not (DEST / "calibration").is_dir()
        or not (DEST / "models").is_dir()
    ):
        raise RuntimeError("Expected DexYCB data layout missing")
    with np.load(labels[0], allow_pickle=False) as sample:
        for key in ["pose_m", "pose_y", "joint_3d"]:
            assert key in sample
        assert sample["pose_m"].shape == (1, 51)
        shapes = {key: list(sample[key].shape) for key in sample.files}
    status(
        "complete",
        downloaded_bytes=TOTAL,
        subject_count=len(list(DEST.glob("2020*-subject-*"))),
        label_file_count=len(labels),
        sample_file=str(labels[0]),
        sample_shapes=shapes,
        archive_sha256=(DEST / "archive_sha256.txt").read_text().split()[0],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
