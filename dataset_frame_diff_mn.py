import os
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import VideoReader
import warnings

warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")


class HDF5VideoDataset(Dataset):
    """
    Splits each video into non-overlapping segments of `sequence_length`
    and resizes diff-frames to `spatial_size`.

    Output per sample:
        video_id   : str
        rgb_emb    : (sequence_length, embed_dim)
        frame_diff : (sequence_length-1, 3, H_resize, W_resize)
        labels     : (C,)
    """

    def __init__(
        self,
        clip_embeddings_dir,
        frame_diff_videos_dir,
        sequence_length=2,
        spatial_size=(224, 224),
        transform=None,
    ):
        super().__init__()
        self.hdf5_path = clip_embeddings_dir
        self.frame_diff_videos_dir = frame_diff_videos_dir
        self.sequence_length = sequence_length
        self.spatial_size = spatial_size
        self.transform = transform

        # Pre-compute (video_id, start_idx, seg_len) list
        self.segments = []
        with h5py.File(self.hdf5_path, "r") as f:
            for vid, grp in f["trimmed_videos"].items():
                T = grp["embeddings"].shape[0]
                if T == 0:
                    continue
                start = 0
                while start < T:
                    seg_len = min(self.sequence_length, T - start)
                    self.segments.append((vid, start, seg_len))
                    start += seg_len

    def __len__(self):
        return len(self.segments)

    # ------------ helpers ------------------------------------------------
    @staticmethod
    def _read_video_segment(path: str, start_idx: int, n_frames: int) -> torch.Tensor:
        """
        Decode exactly `n_frames` frames starting at `start_idx`.
        Returned shape: (T, 3, H, W) uint8.
        """
        vr = VideoReader(path, "video")
        vr.seek(start_idx)
        frames = []
        for _ in range(n_frames):
            try:
                frames.append(next(vr)["data"])  # already (3, H, W)
            except StopIteration:
                break

        if not frames:
            return torch.zeros((n_frames, 3, 1, 1), dtype=torch.uint8)

        frames = torch.stack(frames)  # (T, 3, H, W)

        # Pad by repeating last frame if too short
        if frames.shape[0] < n_frames:
            pad = frames[-1:].repeat(n_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad], dim=0)
        return frames

    def _resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Bilinear resize to self.spatial_size, return float32 in [0,1].
        Input & output: (T, 3, H, W)
        """
        if frames.shape[2:] == self.spatial_size:
            return frames.to(torch.float32) / 255.0

        frames = frames.to(torch.float32) / 255.0
        return F.interpolate(frames, size=self.spatial_size, mode="bilinear", align_corners=False)

    # --------------------------------------------------------------------
    def __getitem__(self, idx):
        video_id, start_idx, seg_len = self.segments[idx]

        # ---- embeddings (slice only the needed window) ------------------
        with h5py.File(self.hdf5_path, "r") as f:
            grp = f[f"trimmed_videos/{video_id}"]
            emb_ds = grp["embeddings"]
            rgb_seq = torch.from_numpy(emb_ds[start_idx : start_idx + seg_len])
            embed_dim = emb_ds.shape[1]
            labels = torch.from_numpy(grp["labels"][:])

        # pad embeddings
        leftover = self.sequence_length - seg_len
        if leftover > 0:
            pad = rgb_seq[-1:].repeat(leftover, 1) if seg_len > 0 else torch.zeros((leftover, embed_dim))
            rgb_seq = torch.cat([rgb_seq, pad], dim=0)

        if self.transform:
            rgb_seq = self.transform(rgb_seq)

        # ---- frame-diff sequence ---------------------------------------
        fd_len = seg_len - 1 if leftover == 0 else self.sequence_length - 1
        frame_diff_path = os.path.join(self.frame_diff_videos_dir, video_id)
        frame_diff = self._read_video_segment(frame_diff_path, start_idx, fd_len)
        frame_diff = self._resize_frames(frame_diff)  # (T, 3, H_r, W_r)

        return {
            "video_id": video_id,
            "rgb_emb": rgb_seq,
            "frame_diff": frame_diff,
            "labels": labels,
        }


# ----------------------------------------------------------------------
def collate_fn(samples):
    video_ids = [s["video_id"] for s in samples]
    rgb_seq = torch.stack([s["rgb_emb"] for s in samples], dim=0)
    frame_seq = torch.stack([s["frame_diff"] for s in samples], dim=0)
    labels = torch.stack([s["labels"] for s in samples], dim=0)
    return {
        "video_id": video_ids,
        "rgb_emb": rgb_seq,
        "frame_diff": frame_seq,
        "labels": labels,
    }


def check_data_loading(dataloader):
    batch = next(iter(dataloader))
    print("Batch keys:", batch.keys())
    print("video_id:", batch["video_id"])
    print("rgb_emb shape:", batch["rgb_emb"].shape)
    print("frame_diff shape:", batch["frame_diff"].shape)
    print("labels shape:", batch["labels"].shape)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    SEQUENCE_LENGTH = 10
    SPATIAL_SIZE = (224, 224)

    train_hdf5_path = "/mnt/Data/mrt/mammalnet/embeddings/mn_train_clip_vit32.h5"
    frame_diff_videos_dir = "/mnt/Data/enz/mammalnet/frame_diff_videos"

    train_dataset = HDF5VideoDataset(
        clip_embeddings_dir=train_hdf5_path,
        frame_diff_videos_dir=frame_diff_videos_dir,
        sequence_length=SEQUENCE_LENGTH,
        spatial_size=SPATIAL_SIZE,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    check_data_loading(train_loader)
