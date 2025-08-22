# -*- coding: utf-8 -*-
"""
Benchmark optical flow vs. frame differencing on multiple random videos.

This version processes multiple random videos from a directory and computes
mean statistics across all examples.

Example:
    python benchmark.py /path/to/videos --num_videos 5 --raft_model raft --raft_ckpt things --keep_videos --save_results
"""

import os
import cv2
import argparse
import numpy as np
import time
import psutil
import json
import random
from pathlib import Path
from typing import Dict, Any, List
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
def find_video_files(directory: str) -> List[str]:
    """Find all video files in the given directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
    video_files = []

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {directory}")

    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))

    return video_files


def sample_random_videos(video_files: List[str], num_videos: int) -> List[str]:
    """Sample random videos from the list."""
    if num_videos >= len(video_files):
        print(f"Warning: Requested {num_videos} videos but only {len(video_files)} available. Using all videos.")
        return video_files

    return random.sample(video_files, num_videos)


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


def compute_mean_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean statistics across all video results."""
    if not all_results:
        return {}

    # Separate successful results
    successful_results = [r for r in all_results if r["optical_flow_metrics"]["success"] and r["frame_difference_metrics"]["success"]]

    if not successful_results:
        return {"error": "No successful results to compute means from"}

    # Initialize accumulators
    of_metrics_sum = {}
    fd_metrics_sum = {}
    video_info_sum = {}
    quality_metrics_sum = {"optical_flow": {}, "frame_difference": {}}

    # Collect all numeric metrics
    numeric_of_keys = ["processing_time_seconds", "max_memory_mb", "output_file_size_mb", "frames_processed", "fps_processing_rate", "memory_per_frame_mb", "max_gpu_memory_mb"]
    numeric_fd_keys = ["processing_time_seconds", "max_memory_mb", "output_file_size_mb", "frames_processed", "fps_processing_rate", "memory_per_frame_mb"]
    numeric_video_keys = ["width", "height", "fps", "frame_count", "duration_seconds", "file_size_mb"]
    numeric_quality_keys = ["mean_pixel_intensity", "std_pixel_intensity", "non_zero_pixels_ratio"]

    # Sum all metrics
    for result in successful_results:
        of_m = result["optical_flow_metrics"]
        fd_m = result["frame_difference_metrics"]
        vi = result["video_info"]
        qm = result.get("quality_metrics", {})

        for key in numeric_of_keys:
            if key in of_m:
                of_metrics_sum[key] = of_metrics_sum.get(key, 0) + of_m[key]

        for key in numeric_fd_keys:
            if key in fd_m:
                fd_metrics_sum[key] = fd_metrics_sum.get(key, 0) + fd_m[key]

        for key in numeric_video_keys:
            if key in vi:
                video_info_sum[key] = video_info_sum.get(key, 0) + vi[key]

        for method in ["optical_flow", "frame_difference"]:
            if method in qm:
                for key in numeric_quality_keys:
                    if key in qm[method]:
                        quality_metrics_sum[method][key] = quality_metrics_sum[method].get(key, 0) + qm[method][key]

    # Compute means
    n = len(successful_results)
    mean_metrics = {
        "num_videos_processed": len(all_results),
        "num_successful_videos": n,
        "optical_flow_metrics": {key: val / n for key, val in of_metrics_sum.items()},
        "frame_difference_metrics": {key: val / n for key, val in fd_metrics_sum.items()},
        "video_info": {key: val / n for key, val in video_info_sum.items()},
        "quality_metrics": {
            method: {key: val / n for key, val in metrics.items()} for method, metrics in quality_metrics_sum.items() if metrics  # Only include if there are metrics
        },
    }

    return mean_metrics


def print_mean_comparison_report(mean_metrics: Dict[str, Any], video_files: List[str]):
    """Print a comprehensive mean comparison report."""
    print("=" * 80)
    print("MULTI-VIDEO BENCHMARK REPORT (MEAN STATISTICS)")
    print("=" * 80)

    if "error" in mean_metrics:
        print(f"❌ Error: {mean_metrics['error']}")
        return

    print(f"\n📊 DATASET INFORMATION:")
    print(f"  Total videos processed: {mean_metrics['num_videos_processed']}")
    print(f"  Successful videos: {mean_metrics['num_successful_videos']}")
    print(f"  Success rate: {mean_metrics['num_successful_videos'] / mean_metrics['num_videos_processed'] * 100:.1f}%")

    # Mean video information
    vi = mean_metrics["video_info"]
    print(f"\n📹 MEAN VIDEO CHARACTERISTICS:")
    print(f"  Resolution: {vi['width']:.0f}x{vi['height']:.0f}")
    print(f"  Duration: {vi['duration_seconds']:.2f} seconds")
    print(f"  Frame Count: {vi['frame_count']:.0f}")
    print(f"  FPS: {vi['fps']:.2f}")
    print(f"  File Size: {vi['file_size_mb']:.2f} MB")

    # Processing comparison
    of_m = mean_metrics["optical_flow_metrics"]
    fd_m = mean_metrics["frame_difference_metrics"]

    print(f"\n⏱️  MEAN PROCESSING TIME COMPARISON:")
    speedup = of_m["processing_time_seconds"] / fd_m["processing_time_seconds"]
    faster_method = "Frame Difference" if speedup > 1 else "Optical Flow"
    print(f"  Optical Flow (RAFT): {of_m['processing_time_seconds']:.2f} seconds")
    print(f"  Frame Difference:    {fd_m['processing_time_seconds']:.2f} seconds")
    print(f"  🏆 {faster_method} is {abs(speedup):.2f}x faster on average")

    # Memory usage comparison
    print(f"\n🧠 MEAN MEMORY USAGE COMPARISON:")
    memory_ratio = of_m["max_memory_mb"] / fd_m["max_memory_mb"] if fd_m["max_memory_mb"] > 0 else float("inf")
    efficient_method = "Frame Difference" if memory_ratio > 1 else "Optical Flow"
    print(f"  Optical Flow (CPU RSS): {of_m['max_memory_mb']:.2f} MB (max)")
    print(f"  Frame Difference (CPU RSS): {fd_m['max_memory_mb']:.2f} MB (max)")
    print(f"  🏆 {efficient_method} uses {abs(memory_ratio):.2f}x less CPU memory on average")

    if "max_gpu_memory_mb" in of_m:
        print(f"  Optical Flow (GPU mem): {of_m['max_gpu_memory_mb']:.2f} MB (peak allocated)")

    # Processing rate comparison
    print(f"\n📊 MEAN PROCESSING RATE:")
    print(f"  Optical Flow (RAFT): {of_m['fps_processing_rate']:.2f} FPS processing rate")
    print(f"  Frame Difference:     {fd_m['fps_processing_rate']:.2f} FPS processing rate")

    # Output file sizes
    print(f"\n💾 MEAN OUTPUT FILE SIZES:")
    print(f"  Optical Flow (RAFT): {of_m['output_file_size_mb']:.2f} MB")
    print(f"  Frame Difference:     {fd_m['output_file_size_mb']:.2f} MB")

    # Quality metrics
    qm = mean_metrics.get("quality_metrics", {})
    if qm:
        print(f"\n🎨 MEAN OUTPUT QUALITY METRICS:")
        for method, q in qm.items():
            if q:  # Only show if metrics exist
                method_name = method.replace("_", " ").title()
                print(f"  {method_name}:")
                print(f"    Mean Pixel Intensity: {q['mean_pixel_intensity']:.2f}")
                print(f"    Std Pixel Intensity:  {q['std_pixel_intensity']:.2f}")
                print(f"    Non-zero Pixels Ratio:{q['non_zero_pixels_ratio']:.4f}")

    # List processed videos
    print(f"\n📁 PROCESSED VIDEOS:")
    for i, video_path in enumerate(video_files, 1):
        print(f"  {i:2d}. {Path(video_path).name}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Benchmark RAFT optical flow vs frame difference processing on multiple random videos")
    parser.add_argument("video_directory", help="Directory containing video files")
    parser.add_argument("--num_videos", type=int, default=5, help="Number of random videos to process (default: 5)")
    parser.add_argument("--output_dir", default="./benchmark_output", help="Directory to save output videos and results (default: ./benchmark_output)")
    parser.add_argument("--save_results", action="store_true", help="Save detailed results to JSON file")
    parser.add_argument("--keep_videos", action="store_true", help="Keep generated videos (default: delete after analysis)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible video sampling")

    # RAFT options
    parser.add_argument("--raft_model", default="raft", help="ptlflow model name (default: raft)")
    parser.add_argument("--raft_ckpt", default="things", help="ptlflow checkpoint (default: things)")
    parser.add_argument("--disable_cudnn", action="store_true", help="Disable cuDNN")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")

    # Validate input directory
    if not os.path.exists(args.video_directory):
        print(f"Error: Video directory not found: {args.video_directory}")
        return 1

    # Find video files
    print(f"🔍 Searching for video files in: {args.video_directory}")
    try:
        all_video_files = find_video_files(args.video_directory)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not all_video_files:
        print(f"Error: No video files found in directory: {args.video_directory}")
        return 1

    print(f"📁 Found {len(all_video_files)} video files")

    # Sample random videos
    selected_videos = sample_random_videos(all_video_files, args.num_videos)
    print(f"🎯 Selected {len(selected_videos)} videos for processing")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting multi-video benchmark")
    print(f"📁 Output directory: {output_dir}")

    try:
        # cuDNN toggle
        if args.disable_cudnn:
            cudnn.enabled = False

        # Load RAFT once
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading RAFT model ({args.raft_model}, {args.raft_ckpt}) on {device}...")
        raft_model, device = load_raft(args.raft_model, args.raft_ckpt, device)

        # Process each video
        all_results = []
        for i, video_path in enumerate(selected_videos, 1):
            video_name = Path(video_path).stem
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(selected_videos)}: {video_name}")
            print(f"{'='*60}")

            # Output paths
            of_output = output_dir / f"{video_name}_optical_flow_raft.mp4"
            fd_output = output_dir / f"{video_name}_frame_diff.mp4"

            try:
                # Video info
                print("📹 Analyzing video...")
                video_info = get_video_info(video_path)

                # RAFT optical flow
                print("🌊 Running optical flow (RAFT) processing...")
                of_metrics = compute_optical_flow_benchmark(video_path, str(of_output), raft_model=raft_model, device=device)

                # Frame differencing
                print("🔄 Running frame difference processing...")
                fd_metrics = compute_frame_difference_benchmark(video_path, str(fd_output))

                # Quality metrics
                print("🎨 Analyzing output quality...")
                quality_metrics = analyze_output_quality(str(of_output), str(fd_output))

                # Store results
                result = {
                    "video_path": video_path,
                    "video_info": video_info,
                    "optical_flow_metrics": of_metrics,
                    "frame_difference_metrics": fd_metrics,
                    "quality_metrics": quality_metrics,
                }
                all_results.append(result)

                # Print individual result summary
                success_status = "✅" if of_metrics["success"] and fd_metrics["success"] else "❌"
                print(f"{success_status} Processing time: OF={of_metrics.get('processing_time_seconds', 'N/A'):.2f}s, FD={fd_metrics.get('processing_time_seconds', 'N/A'):.2f}s")

                # Clean up videos unless requested to keep
                if not args.keep_videos:
                    for vp in [of_output, fd_output]:
                        if vp.exists():
                            vp.unlink()

            except Exception as e:
                print(f"❌ Failed to process {video_name}: {e}")
                # Store failed result
                result = {
                    "video_path": video_path,
                    "video_info": {"error": str(e)},
                    "optical_flow_metrics": {"success": False, "error": str(e)},
                    "frame_difference_metrics": {"success": False, "error": str(e)},
                    "quality_metrics": {},
                }
                all_results.append(result)

        # Compute mean statistics
        print(f"\n🧮 Computing mean statistics across all videos...")
        mean_metrics = compute_mean_metrics(all_results)

        # Print mean report
        print_mean_comparison_report(mean_metrics, selected_videos)

        # Save results (optional)
        if args.save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save individual results
            individual_results_file = output_dir / f"individual_results_{timestamp}.json"
            individual_results = {
                "individual_results": all_results,
                "processing_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "video_directory": str(args.video_directory),
                    "num_videos_requested": args.num_videos,
                    "num_videos_processed": len(selected_videos),
                    "raft_model": args.raft_model,
                    "raft_ckpt": args.raft_ckpt,
                    "seed": args.seed,
                },
            }

            with open(individual_results_file, "w") as f:
                json.dump(individual_results, f, indent=2)
            print(f"💾 Individual results saved to: {individual_results_file}")

            # Save mean results
            mean_results_file = output_dir / f"mean_results_{timestamp}.json"
            mean_results = {
                "mean_metrics": mean_metrics,
                "processing_info": individual_results["processing_info"],
                "selected_videos": selected_videos,
            }

            with open(mean_results_file, "w") as f:
                json.dump(mean_results, f, indent=2)
            print(f"💾 Mean results saved to: {mean_results_file}")

        if args.keep_videos:
            print(f"\n📁 Output videos saved in: {output_dir}")
            print(f"🧹 Individual videos kept as requested")
        else:
            print(f"\n🧹 Temporary videos cleaned up")

        return 0

    except Exception as e:
        print(f"❌ Multi-video benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
