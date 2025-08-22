import os
import glob
import gc
import warnings
import argparse
import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Streaming video reader (lazy decode)
from torchvision.io import VideoReader

# PIL for robust image conversion
from PIL import Image

# Try psutil for memory monitoring; fall back to /proc/meminfo on Linux
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Your model
from models.student_model_frame_diff import FrameDiffStudentModel


# ----------------------------
# Memory helpers & exceptions
# ----------------------------


class LowMemoryError(RuntimeError):
    pass


def _available_gb_fallback() -> float:
    """
    Fallback for machines without psutil (Linux only).
    Reads MemAvailable from /proc/meminfo and returns GB.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    kb = float(parts[1])
                    return kb / (1024**2)
    except Exception:
        pass
    # If we cannot detect, return a big number so we don't trigger false positives.
    return 1e9


def available_gb() -> float:
    if psutil is not None:
        return psutil.virtual_memory().available / (1024**3)
    return _available_gb_fallback()


def memory_guard(min_free_gb: float):
    if available_gb() < min_free_gb:
        raise LowMemoryError(f"Low RAM: {available_gb():.2f} GB available")


def safe_empty_cache():
    gc.collect()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------
# Dataset
# ----------------------------


class FrameDiffVideoDataset(Dataset):
    """
    Minimal dataset that yields full paths to video files (diff frames).
    """

    def __init__(self, frame_diff_videos_dir: str):
        super().__init__()
        pattern = os.path.join(frame_diff_videos_dir, "**", "*.*")  # adjust extension filter as needed
        self.video_paths = [p for p in glob.iglob(pattern, recursive=True) if os.path.isfile(p)]
        self.video_paths.sort()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return self.video_paths[idx]


# ----------------------------
# Optional helper (unused but kept)
# ----------------------------


def latest_checkpoint(ckpt_dir: str) -> str:
    """
    Returns the path to the latest checkpoint in ckpt_dir,
    assuming files are named like student_epoch_1.pth, etc.
    """
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in {ckpt_dir}.")

    def epoch_num(fname):
        base = os.path.splitext(fname)[0]
        return int(base.split("_")[-1]) if base.split("_")[-1].isdigit() else -1

    ckpts.sort(key=lambda x: epoch_num(x))
    return os.path.join(ckpt_dir, ckpts[-1])


# ----------------------------
# Frame normalization utilities
# ----------------------------


def to_chw_uint8(frame: torch.Tensor) -> torch.Tensor:
    """
    Normalize a frame tensor to CHW uint8.
    Accepts HxW, HxWxC, or CxHxW (various backends of VideoReader).
    """
    if not isinstance(frame, torch.Tensor):
        raise TypeError("Frame is not a torch.Tensor")

    if frame.ndim == 2:
        # (H, W) -> add channel dim
        frame = frame.unsqueeze(0)  # (1, H, W)
    elif frame.ndim == 3:
        H, W = frame.shape[-3], frame.shape[-2]
        # Heuristics:
        # - If first dim looks like channel (1/3/4), leave as CHW
        # - Else if last dim looks like channel (1/3/4), permute to CHW
        # - Else fallback: if last dim < 8, treat as channel-last
        if frame.shape[0] in (1, 3, 4) and frame.shape[2] not in (1, 3, 4):
            # already CHW
            pass
        elif frame.shape[2] in (1, 3, 4):
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
        else:
            # Ambiguous layout: prefer HWC if plausible
            if frame.shape[2] < 8:
                frame = frame.permute(2, 0, 1)
            elif frame.shape[0] < 8:
                # likely already CHW
                pass
            else:
                # As a last resort, assume HWC
                frame = frame.permute(2, 0, 1)
    else:
        raise ValueError(f"Unexpected frame shape {tuple(frame.shape)}")

    # Ensure uint8 range for PIL
    if frame.dtype != torch.uint8:
        # Some backends can output other dtypes; normalize/clamp if needed
        frame = frame.clamp(0, 255).to(torch.uint8)

    return frame  # CHW, uint8


def chw_to_pil_rgb(frame_chw_u8: torch.Tensor) -> Image.Image:
    """
    Convert CHW uint8 tensor to a PIL.Image in RGB (guaranteed 3 channels after convert).
    """
    if frame_chw_u8.ndim != 3:
        raise ValueError(f"Expected CHW, got shape {tuple(frame_chw_u8.shape)}")
    # CHW -> HWC numpy
    arr = frame_chw_u8.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    # Force RGB so CLIP preprocess (3-channel) is satisfied, even for grayscale inputs
    return img.convert("RGB")


# ----------------------------
# Streaming decode + transform
# ----------------------------


def read_and_transform_video_streaming(
    video_path: str,
    device: torch.device,
    clip_transform,
    chunk_size: int = 32,
    min_free_gb: float = 1.0,
):
    """
    Lazily decode frames using torchvision.io.VideoReader and yield transformed chunks.
    Each yielded tensor is (chunk, C, H, W). Uses a low-RAM guard between steps.
    """
    vr = VideoReader(video_path, "video")

    frames_processed = []
    for pkt in vr:
        # Check memory before doing more work
        memory_guard(min_free_gb)

        frame = pkt["data"]  # could be HWC or CHW depending on backend
        try:
            frame_chw = to_chw_uint8(frame)
            pil_img = chw_to_pil_rgb(frame_chw)
            transformed = clip_transform(pil_img)  # tensor (C, H, W)
            frames_processed.append(transformed)
        except Exception as e:
            warnings.warn(f"Skipping a frame in {os.path.basename(video_path)} due to conversion error: {e}")
            continue

        if len(frames_processed) == chunk_size:
            batch = torch.stack(frames_processed, dim=0).to(device)  # (chunk, C, H, W)
            frames_processed.clear()
            yield batch
            del batch, frame, frame_chw, pil_img, transformed
            safe_empty_cache()
        else:
            # Release temporaries
            del frame, frame_chw, pil_img, transformed

    # Tail
    if frames_processed:
        batch = torch.stack(frames_processed, dim=0).to(device)
        frames_processed.clear()
        yield batch
        del batch
        safe_empty_cache()


# ----------------------------
# Incremental writer
# ----------------------------


def process_and_write_video_incremental(
    video_path: str,
    model: torch.nn.Module,
    clip_transform,
    device: torch.device,
    h5f: h5py.File,
    chunk_size: int = 32,
    min_free_gb: float = 1.0,
    compression: str = "lzf",
):
    """
    Stream a video, run the model in chunks, and write embeddings incrementally
    to an extendable dataset in h5f. Returns final (T, D) shape.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    group = h5f.require_group(video_id)

    # If already processed, skip (resumable runs)
    if "embeddings" in group:
        dset = group["embeddings"]
        print(f"[{video_id}] already exists, shape={dset.shape}. Skipping.")
        return tuple(dset.shape)

    dset = None
    embed_dim = None
    total_frames = 0

    for chunk_frames in read_and_transform_video_streaming(
        video_path=video_path,
        device=device,
        clip_transform=clip_transform,
        chunk_size=chunk_size,
        min_free_gb=min_free_gb,
    ):
        # Model expects (B, T, 3, H, W)
        inp = chunk_frames.unsqueeze(0)  # (1, chunk, C, H, W)

        with torch.no_grad():
            out, _, _ = model(inp)  # (1, chunk, D)
            embeddings = out.squeeze(0).cpu()  # (chunk, D)

        if embed_dim is None:
            embed_dim = embeddings.shape[1]
            # Create extendable dataset, chunked. Compression optional but "lzf" is fast.
            dset = group.create_dataset(
                "embeddings",
                shape=(0, embed_dim),
                maxshape=(None, embed_dim),
                chunks=(max(1, min(chunk_size, 1024)), embed_dim),
                dtype="float32",
                compression=compression if compression else None,
            )

        # Guard memory before resizing/writing
        memory_guard(min_free_gb)

        # Extend and write
        old_n = dset.shape[0]
        new_n = old_n + embeddings.shape[0]
        dset.resize((new_n, embed_dim))
        dset[old_n:new_n, :] = embeddings.numpy().astype("float32")
        total_frames = new_n

        # Flush to ensure durability
        h5f.flush()

        # Cleanup
        del inp, chunk_frames, embeddings, out
        safe_empty_cache()

    if dset is None:
        # No decodable frames; create empty dataset for bookkeeping
        group.create_dataset("embeddings", shape=(0, 0), maxshape=(None, 0), dtype="float32")
        print(f"[{video_id}] shape=(0, 0) => saved (no decodable frames).")
        return (0, 0)

    print(f"[{video_id}] shape=({total_frames}, {dset.shape[1]}) => saved.")
    return tuple(dset.shape)


# ----------------------------
# Main
# ----------------------------


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # === Dataset and DataLoader ===
    dataset = FrameDiffVideoDataset(args.frame_diff_videos_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # === Model ===
    model = FrameDiffStudentModel(
        clip_model_name=args.clip_model_name,
        device=str(device),
        num_classes=args.num_classes,
    ).to(device)
    model = torch.nn.DataParallel(model)

    # === Load checkpoint ===
    ckpt_path = os.path.join(args.checkpoint_dir, "student_best.pth")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # === Transforms (from FrameDiffStudentModel) ===
    clip_transform = transforms.Compose(model.module.preprocess.transforms)

    # === HDF5 open (append) & optional overwrite ===
    if os.path.exists(args.output_h5_path) and not args.resume and args.overwrite:
        os.remove(args.output_h5_path)

    os.makedirs(os.path.dirname(args.output_h5_path), exist_ok=True)

    print(f"Starting inference. Min free RAM threshold: {args.min_free_gb:.2f} GB")
    print(f"Initial available RAM: {available_gb():.2f} GB")

    processed = 0
    skipped_low_ram = 0
    errors = 0

    # Use "a" mode to allow resuming and incremental flushing
    with h5py.File(args.output_h5_path, "a", libver="latest") as h5f:
        for batch_paths in dataloader:
            for video_path in batch_paths:
                video_path = video_path.strip()
                video_id = os.path.splitext(os.path.basename(video_path))[0]

                # Resume behavior: skip if group already present
                if args.resume and video_id in h5f:
                    print(f"[{video_id}] already exists in HDF5. Skipping (resume).")
                    continue

                try:
                    process_and_write_video_incremental(
                        video_path=video_path,
                        model=model,
                        clip_transform=clip_transform,
                        device=device,
                        h5f=h5f,
                        chunk_size=args.chunk_size,
                        min_free_gb=args.min_free_gb,
                        compression=args.h5_compression,
                    )
                    processed += 1

                except LowMemoryError as e:
                    warnings.warn(f"Skipping {video_id} due to low RAM: {e}")
                    grp = h5f.require_group(video_id)
                    grp.attrs["skipped_low_ram"] = True
                    h5f.flush()
                    skipped_low_ram += 1
                    safe_empty_cache()
                    continue

                except Exception as e:
                    warnings.warn(f"Error on {video_id}: {e}. Moving on.")
                    grp = h5f.require_group(video_id)
                    grp.attrs["error"] = str(e)
                    h5f.flush()
                    errors += 1
                    safe_empty_cache()
                    continue

                # Clean between videos
                safe_empty_cache()

    print(f"Inference complete! Frame_diff embeddings saved to: {args.output_h5_path}\nProcessed: {processed} | Skipped(low RAM): {skipped_low_ram} | Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frame_diff-only CLIP embeddings with a trained FrameDiffStudentModel (MoCLIP) using streaming decode and incremental HDF5 writing."
    )

    # Paths
    parser.add_argument(
        "--frame-diff-videos-dir",
        type=str,
        default="dataset/frame_diffs",
        help="Root directory containing diff-frame videos.",
    )
    parser.add_argument(
        "--output-h5-path",
        type=str,
        default="dataset/embeddings/frame_diff_embeddings.h5",
        help="Destination .h5 file for embeddings.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/20250328-003544",
        help="Folder with the trained student checkpoint 'student_best.pth'.",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="ViT-B/32",
        help="Name of the CLIP visual backbone (same as used in training).",
    )

    # Dataloader settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Videos per inference batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker processes (0 is safest for streaming).",
    )

    # Model metadata
    parser.add_argument(
        "--num-classes",
        type=int,
        default=140,
        help="Number of classes the student was trained for.",
    )

    # Chunking / memory
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Frames per chunk sent to the model (lower => less GPU/RAM usage).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.5,
        help="If available RAM (GB) drops below this, skip the current video.",
    )

    # HDF5 behavior
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos already present in the HDF5 file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If output file exists and --resume is not set, delete it before writing.",
    )
    parser.add_argument(
        "--h5-compression",
        type=str,
        default="",
        choices=["", "lzf", "gzip"],
        help="Compression for HDF5 datasets: '' (none), 'lzf' (fast), or 'gzip' (smaller but slower).",
    )

    args = parser.parse_args()
    main(args)
