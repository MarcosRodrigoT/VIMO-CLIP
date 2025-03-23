import os
import glob
import torch
import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image

from models.student_model import FlowStudentModel


class FlowVideoDataset(Dataset):
    """
    A minimal dataset that returns the full path to each video file (flow frames).
    """

    def __init__(self, flow_videos_dir):
        super().__init__()
        pattern = os.path.join(flow_videos_dir, "**", "*.*")  # Adjust extension filter as needed
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


def read_and_transform_video(video_path, device, clip_transform):
    """
    Reads the video frames from disk and applies the transform that FlowStudentModel expects.
    Returns a tensor of shape (T, C, H, W).
    """
    video, _, _ = read_video(video_path, pts_unit="sec")  # shape: (T, H, W, C)
    video = video.permute(0, 3, 1, 2)  # => (T, C, H, W)

    frames_processed = []
    for frame_tensor in video:
        pil_img = to_pil_image(frame_tensor)
        transformed = clip_transform(pil_img)
        frames_processed.append(transformed)

    return torch.stack(frames_processed, dim=0).to(device)


def main(flow_videos_dir, output_h5_path, checkpoint_dir, clip_model_name, batch_size, num_workers, num_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We'll store all embeddings in one HDF5 file
    if os.path.exists(output_h5_path):
        os.remove(output_h5_path)  # overwrite if desired

    # === Dataset and DataLoader ===
    dataset = FlowVideoDataset(flow_videos_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # === Model ===
    model = FlowStudentModel(clip_model_name=clip_model_name, device=device, num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # === Load latest checkpoint ===
    ckpt_path = latest_checkpoint(checkpoint_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # === Transforms (from FlowStudentModel) ===
    clip_transform = transforms.Compose(model.module.preprocess.transforms)

    # === Inference & Save to .h5 ===
    # We'll store one group per video in the .h5 file
    with h5py.File(output_h5_path, "w") as h5f:
        with torch.no_grad():
            for batch_paths in dataloader:
                for video_path in batch_paths:
                    video_path = video_path.strip()
                    video_id = os.path.splitext(os.path.basename(video_path))[0]

                    # Read and transform frames => (T, C, H, W)
                    frames = read_and_transform_video(video_path, device, clip_transform)

                    # FlowStudentModel expects (B, T, 3, H, W)
                    frames = frames.unsqueeze(0)  # => (1, T, C, H, W)
                    embeddings, _, _ = model(frames)  # => (1, T, embed_dim), (1, num_classes)

                    embeddings = embeddings.squeeze(0).cpu().numpy()  # => (T, embed_dim)

                    # Create group for this video
                    group = h5f.create_group(video_id)
                    group.create_dataset("embeddings", data=embeddings)

                    print(f"[{video_id}] shape={embeddings.shape} => saved to group '{video_id}'.")

    print(f"Inference complete! Flow embeddings saved to: {output_h5_path}")


if __name__ == "__main__":
    # Hyperparameters
    FLOW_VIDEOS_DIR = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"
    OUTPUT_H5_PATH = "./flow_embeddings.h5"
    CHECKPOINT_DIR = "./checkpoints"
    CLIP_MODEL_NAME = "ViT-B/32"
    BATCH_SIZE = 1  # videos per batch
    NUM_WORKERS = 20
    NUM_CLASSES = 140

    main(
        flow_videos_dir=FLOW_VIDEOS_DIR,
        output_h5_path=OUTPUT_H5_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        clip_model_name=CLIP_MODEL_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
    )
