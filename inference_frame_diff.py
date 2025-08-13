import os
import glob
import torch
import h5py
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image

from models.student_model_frame_diff import FrameDiffStudentModel


class FrameDiffVideoDataset(Dataset):
    """
    A minimal dataset that returns the full path to each video file (diff frames).
    """

    def __init__(self, frame_diff_videos_dir):
        super().__init__()
        pattern = os.path.join(frame_diff_videos_dir, "**", "*.*")  # Adjust extension filter as needed
        self.video_paths = [p for p in glob.iglob(pattern, recursive=True) if os.path.isfile(p)]
        self.video_paths.sort()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return self.video_paths[idx]


def latest_checkpoint(ckpt_dir):
    """
    Returns the path to the latest checkpoint in ckpt_dir,
    assuming files are named like student_epoch_1.pth, etc.
    """
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in {ckpt_dir}.")

    def epoch_num(fname):
        # e.g., "student_epoch_5.pth" => parse out 5
        base = os.path.splitext(fname)[0]
        return int(base.split("_")[-1]) if base.split("_")[-1].isdigit() else -1

    ckpts.sort(key=lambda x: epoch_num(x))
    return os.path.join(ckpt_dir, ckpts[-1])


def read_and_transform_video_chunked(video_path, device, clip_transform, chunk_size=32):
    """
    Reads the video frames from disk and applies the transform that FrameDiffStudentModel expects.
    Returns a generator that yields chunks of transformed frames.
    Each chunk has shape (chunk_size, C, H, W) or smaller for the last chunk.
    """
    video, _, _ = read_video(video_path, pts_unit="sec")  # shape: (T, H, W, C)
    video = video.permute(0, 3, 1, 2)  # => (T, C, H, W)

    total_frames = video.shape[0]

    for start_idx in range(0, total_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_frames = video[start_idx:end_idx]

        frames_processed = []
        for frame_tensor in chunk_frames:
            pil_img = to_pil_image(frame_tensor)
            transformed = clip_transform(pil_img)
            frames_processed.append(transformed)

        yield torch.stack(frames_processed, dim=0).to(device)


def process_video_in_chunks(video_path, model, clip_transform, device, chunk_size=32):
    """
    Process a video in chunks and return the concatenated embeddings.
    """
    all_embeddings = []

    for chunk_frames in read_and_transform_video_chunked(video_path, device, clip_transform, chunk_size):
        # FrameDiffStudentModel expects (B, T, 3, H, W)
        chunk_frames = chunk_frames.unsqueeze(0)  # => (1, chunk_size, C, H, W)

        with torch.no_grad():
            embeddings, _, _ = model(chunk_frames)  # => (1, chunk_size, embed_dim)
            embeddings = embeddings.squeeze(0).cpu()  # => (chunk_size, embed_dim)
            all_embeddings.append(embeddings)

    # Concatenate all chunks
    return torch.cat(all_embeddings, dim=0).numpy()  # => (T, embed_dim)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We'll store all embeddings in one HDF5 file
    if os.path.exists(args.output_h5_path):
        os.remove(args.output_h5_path)  # overwrite if desired

    # === Dataset and DataLoader ===
    dataset = FrameDiffVideoDataset(args.frame_diff_videos_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # === Model ===
    model = FrameDiffStudentModel(clip_model_name=args.clip_model_name, device=device, num_classes=args.num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # === Load latest checkpoint ===
    ckpt_path = os.path.join(args.checkpoint_dir, "student_best.pth")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # === Transforms (from FrameDiffStudentModel) ===
    clip_transform = transforms.Compose(model.module.preprocess.transforms)

    # === Inference & Save to .h5 ===
    # We'll store one group per video in the .h5 file
    with h5py.File(args.output_h5_path, "w") as h5f:
        for batch_paths in dataloader:
            for video_path in batch_paths:
                video_path = video_path.strip()
                video_id = os.path.splitext(os.path.basename(video_path))[0]

                # Process video in chunks
                embeddings = process_video_in_chunks(video_path, model, clip_transform, device, chunk_size=args.chunk_size)

                # Create group for this video
                group = h5f.create_group(video_id)
                group.create_dataset("embeddings", data=embeddings)

                print(f"[{video_id}] shape={embeddings.shape} => saved to group '{video_id}'.")

    print(f"Inference complete! Frame_diff embeddings saved to: {args.output_h5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frame_diff-only CLIP embeddings with a trained FrameDiffStudentModel (MoCLIP)")

    # Paths
    parser.add_argument("--frame-diff-videos-dir", type=str, default="dataset/frame_diffs", help="Root directory containing diff-frame videos.")
    parser.add_argument("--output-h5-path", type=str, default="dataset/embeddings/frame_diff_embeddings.h5", help="Destination .h5 file for embeddings.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/20250328-003544", help="Folder with the trained student checkpoint 'student_best.pth'.")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32", help="Name of the CLIP visual backbone (same as used in training).")

    # Dataloader settings
    parser.add_argument("--batch-size", type=int, default=1, help="Videos per inference batch.")
    parser.add_argument("--num-workers", type=int, default=20, help="PyTorch DataLoader worker processes.")

    # Model metadata
    parser.add_argument("--num-classes", type=int, default=140, help="Number of classes the student was trained for.")

    # New argument for chunk processing
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of frames to process at once (lower = less memory usage).")

    args = parser.parse_args()

    main(args)
