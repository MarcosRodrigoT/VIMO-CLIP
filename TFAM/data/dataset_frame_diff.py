import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def sparse_sampling(embeddings, num_frames):
    total_frames = embeddings.shape[0]
    if total_frames > num_frames:
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        embeddings = embeddings[indices]
    return embeddings


class HDF5VideoDataset(Dataset):
    def __init__(self, hdf5_path, frame_diff_path, transform=None, num_frames=None, max_frames=None):
        self.hdf5_path = hdf5_path
        self.frame_diff_path = frame_diff_path
        self.transform = transform
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.keys = []
        self.frame_diff_keys = []

        with h5py.File(hdf5_path, "r") as f:
            all_keys = list(f.keys())
            if self.max_frames:
                filtered_keys = []
                for key in all_keys:
                    if f[key]["embeddings"].shape[0] < self.max_frames:
                        filtered_keys.append(key)
                self.keys = filtered_keys
            else:
                self.keys = all_keys

        with h5py.File(frame_diff_path, "r") as f:
            all_frame_diff_keys = list(f.keys())
            if self.max_frames:
                filtered_frame_diff_keys = []
                for key in all_frame_diff_keys:
                    if f[key]["embeddings"].shape[0] < self.max_frames:
                        filtered_frame_diff_keys.append(key)
                self.frame_diff_keys = filtered_frame_diff_keys
            else:
                self.frame_diff_keys = all_frame_diff_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            video_id = self.keys[idx]
            group = f[video_id]
            embeddings = torch.from_numpy(group["embeddings"][:])
            labels = torch.from_numpy(group["labels"][:])

            if self.num_frames:
                embeddings = sparse_sampling(embeddings, self.num_frames)

            if self.transform:
                embeddings = self.transform(embeddings)

        with h5py.File(self.frame_diff_path, "r") as f:
            # note: your frame_diff keys might not match IDs exactly, so adjust as needed
            frame_diff_id = self.keys[idx].split(".")[0]
            group = f[frame_diff_id]
            frame_diff_embeddings = torch.from_numpy(group["embeddings"][:])
            if self.num_frames:
                frame_diff_embeddings = sparse_sampling(frame_diff_embeddings, self.num_frames)
            if self.transform:
                frame_diff_embeddings = self.transform(frame_diff_embeddings)

        return {"video_id": video_id, "embeddings": embeddings.float(), "frame_diff_embeddings": frame_diff_embeddings.float(), "labels": labels, "total_frames": embeddings.shape[0]}


def collate_fn_pad(batch):
    """
    Collate function that pads variable-length sequences for both RGB and frame_diff,
    and returns distinct boolean masks.
    """
    embeddings = [item["embeddings"] for item in batch]  # RGB
    frame_diff_embeddings = [item["frame_diff_embeddings"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # 1) Pad
    padded_rgb = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)  # (B, T_rgb_max, D)
    padded_frame_diff = torch.nn.utils.rnn.pad_sequence(frame_diff_embeddings, batch_first=True)  # (B, T_frame_diff_max, D)

    # 2) Build mask for each stream
    #   True = real token, False = pad
    lens_rgb = torch.tensor([x.shape[0] for x in embeddings])
    lens_frame_diff = torch.tensor([x.shape[0] for x in frame_diff_embeddings])

    T_rgb_max = padded_rgb.size(1)
    T_frame_diff_max = padded_frame_diff.size(1)

    mask_rgb = torch.arange(T_rgb_max).expand(len(lens_rgb), T_rgb_max)
    mask_rgb = mask_rgb < lens_rgb.unsqueeze(1)
    # shape => (B, T_rgb_max)  (True where it's a real frame)

    mask_frame_diff = torch.arange(T_frame_diff_max).expand(len(lens_frame_diff), T_frame_diff_max)
    mask_frame_diff = mask_frame_diff < lens_frame_diff.unsqueeze(1)
    # shape => (B, T_frame_diff_max)

    return {
        "video_id": [item["video_id"] for item in batch],
        "embeddings": padded_rgb,
        "frame_diff_embeddings": padded_frame_diff,
        "labels": labels,
        "mask_rgb": mask_rgb,  # shape (B, T_rgb_max)
        "mask_frame_diff": mask_frame_diff,  # shape (B, T_frame_diff_max)
    }


################################################################
# Comprobar si los datos se cargan correctamente
################################################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #  Hiperparámetros
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Crear dataset y dataloader
    train_dataset = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_train_clip_vit32.h5"
    # val_dataset = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_val_clip_vit32.h5"
    frame_diff_dataset = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/frame_diff_embeddings_dataset1_si_MLP.h5"

    # Inicializar el dataset y DataLoader
    train_dataset = HDF5VideoDataset(train_dataset, frame_diff_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad, num_workers=NUM_WORKERS)

    # DataLoader con padding
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad, num_workers=NUM_WORKERS)

    # Comprobar si los datos se cargan correctamente
    def check_data_loading(dataloader):
        data_iter = iter(dataloader)
        batch = next(data_iter)

        # Imprimir las claves y los valores de cada parte del batch
        print(f"Claves del batch: {batch.keys()}")
        print(f"video_id: {batch['video_id']}")
        print(f"embeddings: {batch['embeddings'].shape}")
        print(f"frame_diff_embeddings: {batch['frame_diff_embeddings'].shape}")
        # print(f"labels: {batch['labels']}")
        print(f"lengths: {batch['lengths']}")

    # Llamar a la función para verificar si los datos se cargan correctamente
    check_data_loading(dataloader)
