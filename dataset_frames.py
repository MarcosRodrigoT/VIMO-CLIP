import os
import h5py
import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader


class HDF5VideoDataset(Dataset):
    """
    This dataset splits each video into non-overlapping consecutive segments of length `sequence_length`.
    If the last segment is shorter, it is padded up to `sequence_length`.
    Each segment yields:
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
            sequence_length (int): Number of consecutive frames to sample as one segment
            transform (callable, optional): Optional transform on the RGB embeddings
        """
        super().__init__()
        self.hdf5_path = clip_embeddings_dir
        self.flow_videos_dir = flow_videos_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Build a list of (video_id, start_idx, seg_len) for all segments
        self.segments = []  # will hold (video_id, start_idx, seg_len) tuples
        with h5py.File(self.hdf5_path, "r") as f:
            all_keys = list(f.keys())

            for k in all_keys:
                # "embeddings" => shape (T, embed_dim)
                T = f[k]["embeddings"].shape[0]

                if T == 0:
                    # no frames => skip or do a single fully-padded segment
                    continue

                # We'll chunk the video in steps of 'sequence_length', but the last chunk
                # might have leftover. If leftover < sequence_length, we do a single padded chunk.
                start = 0
                while start < T:
                    remaining = T - start
                    if remaining >= self.sequence_length:
                        seg_len = self.sequence_length
                    else:
                        # leftover < sequence_length => we'll do one padded chunk
                        seg_len = remaining
                    self.segments.append((k, start, seg_len))
                    start += seg_len  # jump forward seg_len

    def __len__(self):
        # The total number of consecutive segments we have
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Return the idx-th segment in self.segments, padded if needed.
        """
        video_id, start_idx, seg_len = self.segments[idx]

        # Load embeddings & labels from HDF5
        with h5py.File(self.hdf5_path, "r") as f:
            group = f[video_id]
            embeddings = torch.from_numpy(group["embeddings"][:])  # (T, embed_dim)
            labels = torch.from_numpy(group["labels"][:])  # (C,)

        T = embeddings.shape[0]
        # Extract the segment => shape (seg_len, embed_dim)
        rgb_seq = embeddings[start_idx : start_idx + seg_len]

        # Pad if seg_len < sequence_length
        leftover = self.sequence_length - seg_len
        if leftover > 0:
            # repeat the last frame or zero
            if seg_len > 0:
                last_frame = rgb_seq[-1:].clone()
                pad = last_frame.repeat(leftover, 1)
            else:
                # seg_len=0 => T was 0, which we'd normally skip above
                # but let's handle edge case anyway
                embed_dim = embeddings.shape[1]
                pad = torch.zeros((leftover, embed_dim))
            rgb_seq = torch.cat([rgb_seq, pad], dim=0)  # now shape => (sequence_length, embed_dim)

        if self.transform:
            rgb_seq = self.transform(rgb_seq)

        # Load optical flow video => (T_flow, C, H, W)
        flow_video_path = os.path.join(self.flow_videos_dir, video_id)
        flow_video, _, _ = io.read_video(flow_video_path, pts_unit="sec")
        flow_video = flow_video.permute(0, 3, 1, 2)  # => (T_flow, C, H, W)

        T_flow = flow_video.shape[0]
        # The corresponding flow segment has length = (seg_len - 1), or (sequence_length - 1) if padded
        flow_seg_len = seg_len - 1
        if leftover > 0:
            # If we padded seg_len => full seg len is self.sequence_length
            flow_seg_len = self.sequence_length - 1

        flow_start = start_idx
        flow_end = start_idx + flow_seg_len  # not inclusive

        # clamp if out of range
        flow_start = min(flow_start, max(T_flow - 1, 0))
        flow_end = min(flow_end, T_flow)

        flow_seq = flow_video[flow_start:flow_end]  # shape => (flow_seg_len, C, H, W)

        needed = flow_seg_len - flow_seq.shape[0]
        if needed > 0:
            # pad flow frames
            if flow_seq.shape[0] > 0:
                last_flow = flow_seq[-1:].clone()  # shape => (1, C, H, W)
                pad_flow = last_flow.repeat(needed, 1, 1, 1)
            else:
                # no frames => create zeros
                C, H, W = flow_video.shape[1], flow_video.shape[2], flow_video.shape[3]
                pad_flow = torch.zeros((needed, C, H, W))
            flow_seq = torch.cat([flow_seq, pad_flow], dim=0)

        return {
            "video_id": video_id,
            "rgb_emb": rgb_seq,  # (sequence_length, embed_dim)
            "flow_frames": flow_seq,  # (sequence_length - 1, C, H, W)
            "labels": labels,  # (C,)
        }


def collate_fn(samples):
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
    train_dataset = HDF5VideoDataset(clip_embeddings_dir=train_hdf5_path, flow_videos_dir=flow_videos_dir, sequence_length=SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Verify the data loading process
    check_data_loading(train_loader)
