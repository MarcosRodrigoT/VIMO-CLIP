#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple


def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Zero/unknown frame count for: {video_path}")
    return n


def compute_indices(count: int, num_frames: int) -> List[int]:
    """
    Evenly spaced indices in [0, count-1], unique and sorted.
    """
    arr = np.linspace(0, max(count - 1, 0), num=num_frames, dtype=int)
    uniq = np.unique(arr)  # guard against duplicates if count < num_frames
    return uniq.tolist()


def read_frame_at(video_path: str, frame_idx: int) -> np.ndarray:
    """
    Seeks to a frame and returns it. Raises if not readable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Set position and read
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    if not ok:
        cap.release()
        raise RuntimeError(f"Failed to seek to frame {frame_idx} in {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} in {video_path}")
    return frame


def save_frames(video_path: str, indices: List[int], out_dir: Path):
    """
    Extract frames at indices from video_path and save them into out_dir
    using <basename>_<k>.jpg naming.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(video_path).stem  # filename without extension

    # Open once for efficiency, but still robust to random access
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for k, idx in enumerate(indices):
        idx = min(max(idx, 0), max(total - 1, 0))  # clamp, just in case
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            cap.release()
            # fallback to single open per index
            frame = read_frame_at(video_path, idx)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                # fallback
                frame = read_frame_at(video_path, idx)

        out_path = out_dir / f"{base}_{k}.jpg"
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")

    cap.release()


def main():
    ap = argparse.ArgumentParser(description="Extract 10 evenly spaced, aligned frames from RGB, flow, and difference videos.")
    ap.add_argument("rgb_video", help="Path to the RGB video")
    ap.add_argument("flow_video", help="Path to the optical-flow video")
    ap.add_argument("diff_video", help="Path to the frame-differencing video")
    ap.add_argument("--num", type=int, default=10, help="Number of frames to extract (default: 10)")
    ap.add_argument(
        "--out_root",
        type=str,
        default=".",
        help="Root directory where the output folder will be created (default: current dir)",
    )
    args = ap.parse_args()

    rgb_path = args.rgb_video
    flow_path = args.flow_video
    diff_path = args.diff_video

    # Determine aligned sampling using the shortest video
    n_rgb = get_frame_count(rgb_path)
    n_flow = get_frame_count(flow_path)
    n_diff = get_frame_count(diff_path)
    min_count = min(n_rgb, n_flow, n_diff)

    indices = compute_indices(min_count, args.num)
    if len(indices) < args.num:
        print(f"[WARN] Only {len(indices)} unique indices available from {min_count} frames. Requested {args.num}.")

    # Output structure: <out_root>/<RGB_basename>/{RGB,flow,difference}
    rgb_base_dirname = Path(rgb_path).stem
    base_out_dir = Path(args.out_root) / rgb_base_dirname
    rgb_out = base_out_dir / "RGB"
    flow_out = base_out_dir / "flow"
    diff_out = base_out_dir / "difference"

    print(f"RGB frames  -> {rgb_out}")
    print(f"Flow frames -> {flow_out}")
    print(f"Diff frames -> {diff_out}")
    print(f"Sampling {len(indices)} indices from 0..{min_count-1}: {indices}")

    # Extract and save
    save_frames(rgb_path, indices, rgb_out)
    save_frames(flow_path, indices, flow_out)
    save_frames(diff_path, indices, diff_out)

    print("Done.")


if __name__ == "__main__":
    main()
