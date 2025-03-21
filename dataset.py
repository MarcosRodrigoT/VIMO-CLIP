import os
import h5py
import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def sparse_sampling(embeddings, num_frames):
    """
    Perform sparse sampling on the embeddings.
    Args:
        embeddings (torch.Tensor): Tensor of shape (T, embed_dim) for T frames.
        num_frames (int): Desired number of frames to sample.

    Returns:
        torch.Tensor: Sampled embeddings, shape (num_frames, embed_dim).
    """
    total_frames = embeddings.shape[0]

    if total_frames > num_frames:
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        embeddings = embeddings[indices]

    return embeddings


class HDF5VideoDataset(Dataset):
    """
    Loads CLIP embeddings from an HDF5 file. Each video is stored as a group:
      - "embeddings": (T, embed_dim) tensor of frame embeddings
      - "labels": (C,) tensor of labels
    Optionally performs sparse sampling of frames and applies a transform.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        transform (callable, optional): Transform to apply to the embeddings.
        num_frames (int, optional): If set, performs sparse sampling to this number of frames.
        max_frames (int, optional): If set, skip dataset videos that exceed this number of frames.
    """

    def __init__(self, clip_embeddings_dir, flow_videos_dir, transform=None, num_frames=None, max_frames=None):
        self.hdf5_path = clip_embeddings_dir
        self.flow_videos_dir = flow_videos_dir
        self.transform = transform
        self.num_frames = num_frames
        self.max_frames = max_frames

        # Collect the keys (video IDs) from the HDF5 file.
        with h5py.File(self.hdf5_path, "r") as f:
            all_keys = list(f.keys())

            # Skip videos if they exceed max_frames
            if self.max_frames is not None:
                filtered_keys = []
                for k in all_keys:
                    # Check the number of frames in the embeddings
                    if f[k]["embeddings"].shape[0] <= self.max_frames:
                        filtered_keys.append(k)
                self.keys = filtered_keys
            else:
                self.keys = all_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Open the file (read‐only) and access the group for the given video.
        with h5py.File(self.hdf5_path, "r") as f:
            video_id = self.keys[idx]
            group = f[video_id]

            # Load embeddings and labels
            embeddings = torch.from_numpy(group["embeddings"][:])  # shape (T, embed_dim)
            labels = torch.from_numpy(group["labels"][:])  # shape (C,)

            # If a specific number of frames is requested, do sparse sampling.
            if self.num_frames:
                embeddings = sparse_sampling(embeddings, self.num_frames)

            # Optionally apply a transform (e.g., normalization) on the embeddings.
            if self.transform:
                embeddings = self.transform(embeddings)

            # Load corresponding optical flow video
            flow_video_path = os.path.join(self.flow_videos_dir, video_id)
            flow_video, _, _ = io.read_video(flow_video_path, pts_unit="sec")  # shape: (T, H, W, C)
            flow_video = flow_video.permute(0, 3, 1, 2)  # (T, C, H, W)

            if self.num_frames and flow_video.size(0) > self.num_frames:
                indices = torch.linspace(0, flow_video.size(0) - 1, self.num_frames).long()
                flow_video = flow_video[indices]

            return {
                "video_id": video_id,
                "embeddings": embeddings,
                "labels": labels,
                "flow_video": flow_video,
                "total_frames": embeddings.shape[0],
            }


def collate_fn_pad(batch):
    """
    Collate function to pad variable-length embeddings sequences.
    Returns a dict with padded embeddings, labels, lengths, and video IDs.
    """
    # Separate components
    embeddings_list = [item["embeddings"] for item in batch]  # list of (T_i, embed_dim)
    flow_videos_list = [item["flow_video"] for item in batch]

    labels = torch.stack([item["labels"] for item in batch])  # shape (B, C)
    lengths = torch.tensor([emb.shape[0] for emb in embeddings_list])  # (B,)

    # Pad embeddings along the time dimension
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True)
    padded_flow_videos = pad_sequence(flow_videos_list, batch_first=True)

    return {
        "video_id": [item["video_id"] for item in batch],
        "embeddings": padded_embeddings,
        "flow_video": padded_flow_videos,
        "labels": labels,
        "lengths": lengths,
    }


def check_data_loading(dataloader):
    """
    Fetches a single batch from the DataLoader and prints shape/type info
    for verification.
    """
    data_iter = iter(dataloader)
    batch = next(data_iter)

    print(f"Batch keys: {batch.keys()}")
    print(f"video_id (list): {batch['video_id']}")
    print(f"embeddings shape: {batch['embeddings'].shape}")
    print(f"flow_video shape: {batch['flow_video'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"lengths: {batch['lengths']}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Paths to HDF5 files containing CLIP embeddings for training/validation
    train_hdf5_path = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_train_clip_vit32.h5"
    flow_videos_dir = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"

    # Create dataset and DataLoader
    train_dataset = HDF5VideoDataset(train_hdf5_path, flow_videos_dir, num_frames=None, max_frames=500)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad, num_workers=NUM_WORKERS)

    # Verify the data loading process
    check_data_loading(train_loader)
