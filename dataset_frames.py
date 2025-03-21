import os
import h5py
import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader


class HDF5VideoDataset(Dataset):
    """
    This dataset randomly picks:
      1) A video from the HDF5 file
      2) A consecutive sequence of 'sequence_length' RGB frames from that video
      3) The corresponding 'sequence_length - 1' optical flow frames
      4) The multi-label ground truth for that video

    Each __getitem__ call yields a single sample containing:
      - "video_id": str
      - "rgb_emb": shape (sequence_length, embed_dim)
      - "flow_frames": shape (sequence_length - 1, C, H, W)
      - "labels": shape (C,)
    """

    def __init__(self, clip_embeddings_dir, flow_videos_dir, sequence_length=2, transform=None):
        """
        Args:
            clip_embeddings_dir (str): Path to HDF5 with RGB CLIP embeddings
            flow_videos_dir (str): Directory with optical flow videos
            sequence_length (int): Number of consecutive RGB frames to sample
            transform (callable, optional): Optional transform on the RGB embeddings
        """
        super().__init__()
        self.hdf5_path = clip_embeddings_dir
        self.flow_videos_dir = flow_videos_dir
        self.sequence_length = sequence_length
        self.transform = transform

        with h5py.File(self.hdf5_path, "r") as f:
            all_keys = list(f.keys())

            # Only keep videos that have at least 'sequence_length' frames
            self.keys = []
            for k in all_keys:
                T = f[k]["embeddings"].shape[0]
                if T >= self.sequence_length:
                    self.keys.append(k)

    def __len__(self):
        # Typically, we set dataset size == number of videos.
        # Each __getitem__ => one random sequence from a single video
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Returns a single consecutive sequence of length `sequence_length` from a randomly chosen video.

        The optical flow frames for that sequence will be shape `(sequence_length-1, C, H, W)`.
        """
        video_id = self.keys[idx]

        # Load embeddings & labels from HDF5
        with h5py.File(self.hdf5_path, "r") as f:
            group = f[video_id]
            # shape (T, embed_dim)
            embeddings = torch.from_numpy(group["embeddings"][:])
            labels = torch.from_numpy(group["labels"][:])  # shape (C,)

        # Load optical flow video (T_flow, H, W, C) => (T_flow, C, H, W)
        flow_video_path = os.path.join(self.flow_videos_dir, video_id)
        flow_video, _, _ = io.read_video(flow_video_path, pts_unit="sec")
        flow_video = flow_video.permute(0, 3, 1, 2)  # (T_flow, C, H, W)

        T = embeddings.shape[0]  # e.g. if T=100 for the RGB frames
        T_flow = flow_video.shape[0]  # typically T_flow ~ T-1 if correct

        # 1) Randomly pick the start index 'i' of your sequence in [0 .. T - sequence_length]
        i = torch.randint(0, T - self.sequence_length + 1, size=(1,)).item()

        # 2) Extract the RGB embeddings => shape (sequence_length, embed_dim)
        rgb_seq = embeddings[i : i + self.sequence_length]

        if self.transform:
            rgb_seq = self.transform(rgb_seq)

        # 3) Extract the optical flow frames => shape (sequence_length-1, C, H, W)
        # We want flow frames [i..(i+sequence_length-2)] if T_flow ~ T-1, so we do:
        flow_start = i
        flow_end = i + self.sequence_length - 1  # not inclusive => i..flow_end-1
        # But clamp if T_flow < flow_end
        flow_end = min(flow_end, T_flow)

        flow_seq = flow_video[flow_start:flow_end]
        # If for some reason the flow_seq is too short, we can pad or clamp
        expected_len = self.sequence_length - 1
        if flow_seq.shape[0] < expected_len:
            # Simple clamp approach: repeat last frame
            last = flow_seq[-1].unsqueeze(0).clone()
            needed = expected_len - flow_seq.shape[0]
            fill = last.repeat(needed, 1, 1, 1)
            flow_seq = torch.cat([flow_seq, fill], dim=0)
            # shape => (sequence_length-1, C, H, W)

        # 4) Return the item
        return {
            "video_id": video_id,
            "rgb_emb": rgb_seq,  # (sequence_length, embed_dim)
            "flow_frames": flow_seq,  # (sequence_length - 1, C, H, W)
            "labels": labels,  # (C,)
        }


def collate_fn(samples):
    """
    Collate a list of samples (each with a consecutive sequence of frames).
    Returns:
      {
        "video_id": [list of str],
        "rgb_emb": shape (B, sequence_length, embed_dim),
        "flow_frames": shape (B, sequence_length - 1, C, H, W),
        "labels": shape (B, C),
      }
    """
    video_ids = [s["video_id"] for s in samples]

    rgb_seq = torch.stack([s["rgb_emb"] for s in samples], dim=0)  # (B, seq_len, embed_dim)
    flow_seq = torch.stack([s["flow_frames"] for s in samples], dim=0)  # (B, seq_len-1, C, H, W)
    labels = torch.stack([s["labels"] for s in samples], dim=0)

    return {
        "video_id": video_ids,
        "rgb_emb": rgb_seq,
        "flow_frames": flow_seq,
        "labels": labels,
    }


def check_data_loading(dataloader):
    """
    Grab a single batch and print shape/info
    """
    data_iter = iter(dataloader)
    batch = next(data_iter)

    print("Batch keys:", batch.keys())
    print("video_id:", batch["video_id"])
    print("rgb_emb shape:", batch["rgb_emb"].shape)  # (B, seq_len, embed_dim)
    print("flow_frames shape:", batch["flow_frames"].shape)  # (B, seq_len-1, C, H, W)
    print("labels shape:", batch["labels"].shape)  # (B, C)


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    SEQUENCE_LENGTH = 10

    # Paths to HDF5 files containing CLIP embeddings for training and path to optical flow videos directory
    train_hdf5_path = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_train_clip_vit32.h5"
    flow_videos_dir = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"

    # Create dataset and DataLoader
    train_dataset = HDF5VideoDataset(train_hdf5_path, flow_videos_dir, sequence_length=SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Verify the data loading process
    check_data_loading(train_loader)
