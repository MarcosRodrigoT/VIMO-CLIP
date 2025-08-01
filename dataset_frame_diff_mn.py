import os
import h5py
import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader


class HDF5VideoDataset(Dataset):
    """
    This dataset splits each video into non-overlapping consecutive segments of
    length `sequence_length`. If the last segment is shorter, it is padded up to
    `sequence_length`.

    Each segment yields:
      - "video_id": str
      - "rgb_emb": shape (sequence_length, embed_dim)
      - "diff_frames": shape (sequence_length - 1, C, H, W)
      - "labels": shape (C,)
    """

    def __init__(
        self,
        clip_embeddings_dir,
        frame_diff_videos_dir,
        sequence_length=2,
        transform=None,
    ):
        """
        Args:
            clip_embeddings_dir (str): Path to HDF5 with RGB CLIP embeddings
            frame_diff_videos_dir (str): Directory with frame difference videos
            sequence_length (int): Number of consecutive frames to sample
            transform (callable, optional): Optional transform on the RGB embeddings
        """
        super().__init__()
        self.hdf5_path = clip_embeddings_dir
        self.frame_diff_videos_dir = frame_diff_videos_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # -------------------------------------------------------------
        # Build a list of (video_id, start_idx, seg_len) for all videos
        # -------------------------------------------------------------
        self.segments = []  # (video_id, start_idx, seg_len)
        with h5py.File(self.hdf5_path, "r") as f:
            videos_root = f["trimmed_videos"]  # <── NEW: dive into sub-group

            for video_id, grp in videos_root.items():
                # "embeddings" => shape (T, embed_dim)
                T = grp["embeddings"].shape[0]

                if T == 0:
                    continue  # skip empty videos

                start = 0
                while start < T:
                    remaining = T - start
                    seg_len = (
                        self.sequence_length
                        if remaining >= self.sequence_length
                        else remaining
                    )
                    self.segments.append((video_id, start, seg_len))
                    start += seg_len

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.segments)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        video_id, start_idx, seg_len = self.segments[idx]

        # ----- Load embeddings & labels from HDF5 ----------------------
        with h5py.File(self.hdf5_path, "r") as f:
            group_path = f"trimmed_videos/{video_id}"
            group = f[group_path]
            embeddings = torch.from_numpy(group["embeddings"][:])  # (T, embed_dim)
            labels = torch.from_numpy(group["labels"][:])           # (C,)

        # ----- Slice / pad the embeddings -----------------------------
        rgb_seq = embeddings[start_idx : start_idx + seg_len]
        leftover = self.sequence_length - seg_len
        if leftover > 0:
            if seg_len > 0:
                pad = rgb_seq[-1:].repeat(leftover, 1)
            else:
                pad = torch.zeros((leftover, embeddings.shape[1]))
            rgb_seq = torch.cat([rgb_seq, pad], dim=0)

        if self.transform:
            rgb_seq = self.transform(rgb_seq)

        # ----- Load frame-difference video -----------------------------
        frame_diff_video_path = os.path.join(self.frame_diff_videos_dir, video_id)
        frame_diff_video, _, _ = io.read_video(frame_diff_video_path, pts_unit="sec")
        frame_diff_video = frame_diff_video.permute(0, 3, 1, 2)  # (T_fd, C, H, W)

        # ----- Slice / pad the frame-difference segment ---------------
        frame_diff_seg_len = (
            seg_len - 1 if leftover == 0 else self.sequence_length - 1
        )
        frame_diff_start = start_idx
        frame_diff_end = start_idx + frame_diff_seg_len
        frame_diff_start = min(frame_diff_start, max(frame_diff_video.shape[0] - 1, 0))
        frame_diff_end = min(frame_diff_end, frame_diff_video.shape[0])

        frame_diff_seq = frame_diff_video[frame_diff_start:frame_diff_end]

        needed = frame_diff_seg_len - frame_diff_seq.shape[0]
        if needed > 0:
            if frame_diff_seq.shape[0] > 0:
                pad_frame = frame_diff_seq[-1:].repeat(needed, 1, 1, 1)
            else:
                C, H, W = frame_diff_video.shape[1:4]
                pad_frame = torch.zeros((needed, C, H, W))
            frame_diff_seq = torch.cat([frame_diff_seq, pad_frame], dim=0)

        return {
            "video_id": video_id,
            "rgb_emb": rgb_seq,            # (sequence_length, embed_dim)
            "frame_diff": frame_diff_seq,  # (sequence_length - 1, C, H, W)
            "labels": labels,              # (C,)
        }


# ----------------------------------------------------------------------
def collate_fn(samples):
    video_ids = [s["video_id"] for s in samples]
    rgb_seq = torch.stack([s["rgb_emb"] for s in samples], dim=0)
    frame_diff_seq = torch.stack([s["frame_diff"] for s in samples], dim=0)
    labels = torch.stack([s["labels"] for s in samples], dim=0)
    return {
        "video_id": video_ids,
        "rgb_emb": rgb_seq,
        "frame_diff": frame_diff_seq,
        "labels": labels,
    }


def check_data_loading(dataloader):
    data_iter = iter(dataloader)
    batch = next(data_iter)
    print("Batch keys:", batch.keys())
    print("video_id:", batch["video_id"])
    print("rgb_emb shape:", batch["rgb_emb"].shape)
    print("frame_diff shape:", batch["frame_diff"].shape)
    print("labels shape:", batch["labels"].shape)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    SEQUENCE_LENGTH = 10

    # Paths
    train_hdf5_path = "/mnt/Data/mrt/mammalnet/embeddings/mn_train_clip_vit32.h5"
    frame_diff_videos_dir = "/mnt/Data/enz/mammalnet/frame_diff_videos"

    # Dataset & loader
    train_dataset = HDF5VideoDataset(
        clip_embeddings_dir=train_hdf5_path,
        frame_diff_videos_dir=frame_diff_videos_dir,
        sequence_length=SEQUENCE_LENGTH,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    # Quick sanity-check
    check_data_loading(train_loader)

