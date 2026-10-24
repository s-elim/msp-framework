#!/usr/bin/env python3
"""Downloader and unpacker for LineMOD (LM, BOP Benchmark) with resume & retry."""

import argparse
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HF_BASE_URL = "https://huggingface.co/datasets/bop-benchmark/lm/resolve/main"

FILES = {
    "base": ["lm_base.zip"],
    "models": ["lm_models.zip"],
    "train_real": ["lm_train.zip"],
    "test": ["lm_test_all.zip"],
    "pbr": ["lm_train_pbr.zip"],
}


def download_file(url: str, dest_path: Path, log_file: Path | None = None) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    def log(msg: str):
        print(msg, flush=True)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log(f"Starting download: {dest_path.name} from {url}")

    while True:
        initial_size = temp_path.stat().st_size if temp_path.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if initial_size > 0:
            req.add_header("Range", f"bytes={initial_size}-")

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                if resp.status == 200 and initial_size > 0:
                    initial_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if initial_size > 0 else "wb"

                content_length = resp.headers.get("Content-Length")
                total_size = int(content_length) + initial_size if content_length else None

                log(f"Target size: {total_size / (1024**3):.2f} GB" if total_size else "Size unknown")

                downloaded = initial_size
                last_log_time = time.time()
                start_time = time.time()
                chunk_size = 4 * 1024 * 1024

                with open(temp_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_log_time >= 10:
                            speed_mb = (downloaded - initial_size) / (now - start_time) / (1024 * 1024)
                            pct = (downloaded / total_size * 100) if total_size else 0
                            log(f"{dest_path.name}: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3) if total_size else 0:.2f} GB ({pct:.1f}%) at {speed_mb:.1f} MB/s")
                            last_log_time = now

            if total_size is None or downloaded >= total_size:
                temp_path.rename(dest_path)
                log(f"Completed download: {dest_path.name}")
                return dest_path
            else:
                log(f"Incomplete download ({downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB). Reconnecting to resume...")
                time.sleep(2)
        except Exception as e:
            log(f"Network glitch while fetching {dest_path.name}: {e}. Retrying in 5 seconds...")
            time.sleep(5)


def extract_zip(archive_path: Path, target_dir: Path, log_file: Path | None = None):
    def log(msg: str):
        print(msg, flush=True)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log(f"Extracting {archive_path.name} into {target_dir}...")
    with zipfile.ZipFile(archive_path) as z:
        for m in z.infolist():
            parts = m.filename.split("/", 1)
            if parts[0] in ("lm", "lm_models", "lm_test_all", "lm_train", "lm_train_pbr") and len(parts) > 1 and parts[1]:
                dest = target_dir / parts[1]
            else:
                dest = target_dir / m.filename
            if m.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(z.read(m))
    log(f"Extraction of {archive_path.name} finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["base", "models", "train_real", "test"], choices=["base", "models", "train_real", "test", "pbr", "all"])
    parser.add_argument("--target-dir", type=str, default="/nas/Selim/selim_sarowar/msp_framework/data/lm")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "download_lm.log"

    splits = args.splits
    if "all" in splits:
        splits = ["base", "models", "train_real", "test", "pbr"]

    for split in splits:
        files = FILES[split]
        for f in files:
            url = f"{HF_BASE_URL}/{f}"
            dest = target_dir / f

            # Check if file exists and is valid
            is_valid = False
            if dest.exists():
                if f.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(dest) as z:
                            z.testzip()
                        is_valid = True
                        print(f"Archive {f} already exists and is valid. Skipping download.")
                    except Exception:
                        print(f"Archive {f} is corrupted or incomplete. Renaming to .part to resume.")
                        temp = dest.with_suffix(dest.suffix + ".part")
                        dest.rename(temp)
                else:
                    is_valid = True

            if not is_valid:
                download_file(url, dest, log_file)

            if f.endswith(".zip"):
                extract_zip(dest, target_dir, log_file)

    print("All requested LineMOD splits processed successfully.")


if __name__ == "__main__":
    main()
