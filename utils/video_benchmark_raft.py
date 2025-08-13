# -*- coding: utf-8 -*-
"""
Benchmark optical flow vs. frame differencing.

This version replaces the OpenCV Farnebäck optical flow with RAFT (via ptlflow).
- Keeps your MemoryMonitor for CPU RSS.
- Adds optional GPU memory tracking for RAFT runs.
- Preserves your reporting format and file outputs.

Example:
    python benchmark.py input.mp4 --raft_model raft --raft_ckpt things --keep_videos --save_results
"""

import os
import cv2
import argparse
import numpy as np
import time
import psutil
import json
from pathlib import Path
from typing import Dict, Any
import threading

# --- NEW: torch / ptlflow imports for RAFT ---
import torch
import torch.backends.cudnn as cudnn
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils


# =========================
# Monitoring utilities
# =========================
class MemoryMonitor:
    """Monitor CPU memory usage during processing."""

    def __init__(self):
        self.max_memory_mb = 0
        self.current_memory_mb = 0
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        self.monitoring = True
        self.max_memory_mb = 0
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.max_memory_mb

    def _monitor_memory(self):
        process = psutil.Process(os.getpid())
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                current_mb = memory_info.rss / (1024 * 1024)  # MB
                self.current_memory_mb = current_mb
                self.max_memory_mb = max(self.max_memory_mb, current_mb)
                time.sleep(0.1)
            except Exception:
                break


# =========================
# Video utilities
# =========================
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Extract basic information about the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1e-3:
        fps = 25.0  # robust fallback

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps if fps > 0 else 0.0,
        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
    }
    cap.release()
    return info


# =========================
# RAFT (ptlflow) helpers
# =========================
def load_raft(model_name: str = "raft", ckpt_path: str = "things", device: torch.device = None):
    """
    Load a RAFT model via ptlflow. Default weights: 'things' (as in your earlier code).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cuDNN can be disabled if you need to replicate earlier behavior
    # cudnn.enabled = False  # Uncomment if you need strict determinism or to mirror prior runs

    model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    return model, device


def _flow_to_bgr(flow_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a flow tensor (B, 2, H, W) to a BGR uint8 frame using ptlflow's flow_to_rgb.
    Handles potential extra dims returned by the util.
    """
    # Ensure it's on CPU for conversion
    flow_tensor = flow_tensor.detach().cpu()

    rgb = flow_utils.flow_to_rgb(flow_tensor)
    # rgb can be (B,3,H,W) or (B,1,3,H,W) depending on version
    if rgb.ndim == 5:  # (B,1,3,H,W)
        rgb = rgb[0, 0]  # (3,H,W)
    elif rgb.ndim == 4:  # (B,3,H,W)
        rgb = rgb[0]  # (3,H,W)
    elif rgb.ndim == 3:  # (3,H,W)
        pass
    else:
        raise RuntimeError(f"Unexpected rgb shape from flow_to_rgb: {tuple(rgb.shape)}")

    rgb = rgb.permute(1, 2, 0).numpy()  # HWC, float in [0,1]
    bgr = cv2.cvtColor((255.0 * np.clip(rgb, 0.0, 1.0)).astype("uint8"), cv2.COLOR_RGB2BGR)
    return bgr


# =========================
# Processing paths
# =========================
def compute_optical_flow_benchmark(video_path: str, output_path: str, raft_model=None, device: torch.device = None) -> Dict[str, Any]:
    """
    Compute optical flow with benchmarking using RAFT (ptlflow).
    This replaces the prior Farnebäck implementation.
    """
    memory_monitor = MemoryMonitor()
    start_time = time.time()
    memory_monitor.start_monitoring()

    metrics = {"method": "optical_flow", "backend": "RAFT (ptlflow)", "success": False, "error": None, "frames_processed": 0}

    try:
        if raft_model is None:
            raft_model, device = load_raft()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 25.0

        # Read the first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame")

        height, width = prev_frame.shape[:2]

        # Init writer (color frames for the RAFT flow visualization)
        # Try multiple codecs for compatibility
        codecs_to_try = ["mp4v", "XVID", "MJPG", "X264"]
        out = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
                if out.isOpened():
                    break
                else:
                    out.release()
                    out = None
            except Exception:
                if out:
                    out.release()
                    out = None
                continue
        if out is None:
            raise ValueError(f"Could not initialize VideoWriter with any codec for {output_path}")

        # Build IOAdapter for the current image size
        model_for_adapter = raft_model.module if isinstance(raft_model, torch.nn.DataParallel) else raft_model
        io_adapter = IOAdapter(model_for_adapter, (height, width))

        frames_processed = 0
        torch.cuda.reset_peak_memory_stats(device) if device and device.type == "cuda" else None

        with torch.no_grad():
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break

                # Prepare a two-frame clip in the format RAFT expects via IOAdapter
                pair_np = np.stack([prev_frame, curr_frame], axis=0)  # (2, H, W, C)
                inputs = io_adapter.prepare_inputs(pair_np)
                images = inputs["images"].to(device, non_blocking=True)

                # Inference
                predictions = raft_model({"images": images})
                flow = predictions["flows"]  # expected (B, 2, H, W)

                # Visualization & write
                flow_bgr = _flow_to_bgr(flow)
                out.write(flow_bgr)

                prev_frame = curr_frame
                frames_processed += 1

        cap.release()
        out.release()

        metrics["success"] = True
        metrics["frames_processed"] = frames_processed

    except Exception as e:
        metrics["error"] = str(e)

    # Stop monitoring and collect metrics
    end_time = time.time()
    max_memory_mb = memory_monitor.stop_monitoring()
    metrics.update(
        {
            "processing_time_seconds": end_time - start_time,
            "max_memory_mb": max_memory_mb,
            "output_file_size_mb": os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0,
        }
    )

    if metrics["frames_processed"] > 0:
        metrics["fps_processing_rate"] = metrics["frames_processed"] / metrics["processing_time_seconds"]
        metrics["memory_per_frame_mb"] = max_memory_mb / metrics["frames_processed"]

    # Optional: GPU memory stats (if CUDA)
    if torch.cuda.is_available() and device is not None and device.type == "cuda":
        try:
            metrics["max_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        except Exception:
            pass

    return metrics


def compute_frame_difference_benchmark(video_path: str, output_path: str) -> Dict[str, Any]:
    """Compute frame difference with benchmarking (unchanged)."""
    memory_monitor = MemoryMonitor()
    start_time = time.time()
    memory_monitor.start_monitoring()

    metrics = {"method": "frame_difference", "success": False, "error": None, "frames_processed": 0}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 25.0

        codecs_to_try = ["mp4v", "XVID", "MJPG", "X264"]
        out = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
                if out.isOpened():
                    break
                else:
                    out.release()
                    out = None
            except Exception:
                if out:
                    out.release()
                    out = None
                continue
        if out is None:
            raise ValueError(f"Could not initialize VideoWriter with any codec for {output_path}")

        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frames_processed = 0

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            out.write(frame_diff)
            prev_gray = curr_gray
            frames_processed += 1

        cap.release()
        out.release()
        metrics["success"] = True
        metrics["frames_processed"] = frames_processed

    except Exception as e:
        metrics["error"] = str(e)

    end_time = time.time()
    max_memory_mb = memory_monitor.stop_monitoring()
    metrics.update(
        {
            "processing_time_seconds": end_time - start_time,
            "max_memory_mb": max_memory_mb,
            "output_file_size_mb": os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0,
        }
    )

    if metrics["frames_processed"] > 0:
        metrics["fps_processing_rate"] = metrics["frames_processed"] / metrics["processing_time_seconds"]
        metrics["memory_per_frame_mb"] = max_memory_mb / metrics["frames_processed"]

    return metrics


def analyze_output_quality(optical_flow_path: str, frame_diff_path: str) -> Dict[str, Any]:
    """Analyze basic quality metrics of the output videos (unchanged)."""
    quality_metrics = {}
    if os.path.exists(optical_flow_path):
        cap_of = cv2.VideoCapture(optical_flow_path)
        if cap_of.isOpened():
            ret, frame = cap_of.read()
            if ret:
                mean_intensity = float(np.mean(frame))
                std_intensity = float(np.std(frame))
                quality_metrics["optical_flow"] = {
                    "mean_pixel_intensity": mean_intensity,
                    "std_pixel_intensity": std_intensity,
                    "non_zero_pixels_ratio": float(np.count_nonzero(frame) / frame.size),
                }
            cap_of.release()

    if os.path.exists(frame_diff_path):
        cap_fd = cv2.VideoCapture(frame_diff_path)
        if cap_fd.isOpened():
            ret, frame = cap_fd.read()
            if ret:
                mean_intensity = float(np.mean(frame))
                std_intensity = float(np.std(frame))
                quality_metrics["frame_difference"] = {
                    "mean_pixel_intensity": mean_intensity,
                    "std_pixel_intensity": std_intensity,
                    "non_zero_pixels_ratio": float(np.count_nonzero(frame) / frame.size),
                }
            cap_fd.release()

    return quality_metrics


def print_comparison_report(video_info: Dict[str, Any], of_metrics: Dict[str, Any], fd_metrics: Dict[str, Any], quality_metrics: Dict[str, Any]):
    """Print a comprehensive comparison report (unchanged formatting)."""
    print("=" * 80)
    print("VIDEO PROCESSING BENCHMARK REPORT")
    print("=" * 80)

    # Video information
    print(f"\n📹 VIDEO INFORMATION:")
    print(f"  Resolution: {video_info['resolution']}")
    print(f"  Duration: {video_info['duration_seconds']:.2f} seconds")
    print(f"  Frame Count: {video_info['frame_count']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  File Size: {video_info['file_size_mb']:.2f} MB")

    # Processing comparison
    print(f"\n⏱️  PROCESSING TIME COMPARISON:")
    if of_metrics["success"] and fd_metrics["success"]:
        speedup = of_metrics["processing_time_seconds"] / fd_metrics["processing_time_seconds"]
        faster_method = "Frame Difference" if speedup > 1 else "Optical Flow"
        print(f"  Optical Flow (RAFT): {of_metrics['processing_time_seconds']:.2f} seconds")
        print(f"  Frame Difference:    {fd_metrics['processing_time_seconds']:.2f} seconds")
        print(f"  🏆 {faster_method} is {abs(speedup):.2f}x faster")
    else:
        print(f"  Optical Flow: {'FAILED' if not of_metrics['success'] else f'{of_metrics['processing_time_seconds']:.2f} seconds'}")
        print(f"  Frame Difference: {'FAILED' if not fd_metrics['success'] else f'{fd_metrics['processing_time_seconds']:.2f} seconds'}")

    # Memory usage comparison
    print(f"\n🧠 MEMORY USAGE COMPARISON:")
    if of_metrics["success"] and fd_metrics["success"]:
        memory_ratio = of_metrics["max_memory_mb"] / fd_metrics["max_memory_mb"] if fd_metrics["max_memory_mb"] > 0 else float("inf")
        efficient_method = "Frame Difference" if memory_ratio > 1 else "Optical Flow"
        print(f"  Optical Flow (CPU RSS): {of_metrics['max_memory_mb']:.2f} MB (max)")
        print(f"  Frame Difference (CPU RSS): {fd_metrics['max_memory_mb']:.2f} MB (max)")
        print(f"  🏆 {efficient_method} uses {abs(memory_ratio):.2f}x less CPU memory")
    else:
        print(f"  Optical Flow (CPU RSS): {'N/A' if not of_metrics['success'] else f'{of_metrics['max_memory_mb']:.2f} MB'}")
        print(f"  Frame Difference (CPU RSS): {'N/A' if not fd_metrics['success'] else f'{fd_metrics['max_memory_mb']:.2f} MB'}")

    # Optional GPU memory info
    if "max_gpu_memory_mb" in of_metrics:
        print(f"  Optical Flow (GPU mem): {of_metrics['max_gpu_memory_mb']:.2f} MB (peak allocated)")

    # Processing rate comparison
    print(f"\n📊 PROCESSING RATE:")
    if of_metrics.get("fps_processing_rate") is not None:
        print(f"  Optical Flow (RAFT): {of_metrics['fps_processing_rate']:.2f} FPS processing rate")
    if fd_metrics.get("fps_processing_rate") is not None:
        print(f"  Frame Difference:     {fd_metrics['fps_processing_rate']:.2f} FPS processing rate")

    # Output file sizes
    print(f"\n💾 OUTPUT FILE SIZES:")
    if of_metrics["success"]:
        print(f"  Optical Flow (RAFT): {of_metrics['output_file_size_mb']:.2f} MB")
    if fd_metrics["success"]:
        print(f"  Frame Difference:     {fd_metrics['output_file_size_mb']:.2f} MB")

    # Quality metrics
    if quality_metrics:
        print(f"\n🎨 OUTPUT QUALITY METRICS:")
        for method, q in quality_metrics.items():
            method_name = method.replace("_", " ").title()
            print(f"  {method_name}:")
            print(f"    Mean Pixel Intensity: {q['mean_pixel_intensity']:.2f}")
            print(f"    Std Pixel Intensity:  {q['std_pixel_intensity']:.2f}")
            print(f"    Non-zero Pixels Ratio:{q['non_zero_pixels_ratio']:.4f}")

    # Errors
    if not of_metrics["success"] or not fd_metrics["success"]:
        print(f"\n❌ ERRORS:")
        if not of_metrics["success"]:
            print(f"  Optical Flow: {of_metrics['error']}")
        if not fd_metrics["success"]:
            print(f"  Frame Difference: {fd_metrics['error']}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Benchmark RAFT optical flow vs frame difference processing")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output_dir", default="./benchmark_output", help="Directory to save output videos and results (default: ./benchmark_output)")
    parser.add_argument("--save_results", action="store_true", help="Save detailed results to JSON file")
    parser.add_argument("--keep_videos", action="store_true", help="Keep generated videos (default: delete after analysis)")

    # NEW: RAFT options
    parser.add_argument("--raft_model", default="raft", help="ptlflow model name (default: raft)")
    parser.add_argument("--raft_ckpt", default="things", help="ptlflow checkpoint (default: things)")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN (mirrors your earlier script)")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    video_name = Path(args.video_path).stem
    of_output = output_dir / f"{video_name}_optical_flow_raft.mp4"
    fd_output = output_dir / f"{video_name}_frame_diff.mp4"

    print(f"🚀 Starting benchmark for: {args.video_path}")
    print(f"📁 Output directory: {output_dir}")

    try:
        # cuDNN toggle
        if args.disable_cudnn:
            cudnn.enabled = False

        # Load RAFT once
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raft_model, device = load_raft(args.raft_model, args.raft_ckpt, device)

        # Video info
        print("\n📹 Analyzing video...")
        video_info = get_video_info(args.video_path)

        # RAFT optical flow
        print("🌊 Running optical flow (RAFT) processing...")
        of_metrics = compute_optical_flow_benchmark(args.video_path, str(of_output), raft_model=raft_model, device=device)

        # Frame differencing
        print("🔄 Running frame difference processing...")
        fd_metrics = compute_frame_difference_benchmark(args.video_path, str(fd_output))

        # Quality metrics
        print("🎨 Analyzing output quality...")
        quality_metrics = analyze_output_quality(str(of_output), str(fd_output))

        # Report
        print_comparison_report(video_info, of_metrics, fd_metrics, quality_metrics)

        # Save results (optional)
        if args.save_results:
            results = {
                "video_info": video_info,
                "optical_flow_metrics": of_metrics,
                "frame_difference_metrics": fd_metrics,
                "quality_metrics": quality_metrics,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_video": str(args.video_path),
                "raft_model": args.raft_model,
                "raft_ckpt": args.raft_ckpt,
            }
            results_file = output_dir / f"{video_name}_benchmark_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Detailed results saved to: {results_file}")

        # Clean up videos unless requested to keep
        if not args.keep_videos:
            for vp in [of_output, fd_output]:
                if vp.exists():
                    vp.unlink()
            print(f"\n🧹 Temporary videos cleaned up")
        else:
            print(f"\n📁 Output videos saved:")
            print(f"  Optical Flow (RAFT): {of_output}")
            print(f"  Frame Difference:    {fd_output}")

        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
