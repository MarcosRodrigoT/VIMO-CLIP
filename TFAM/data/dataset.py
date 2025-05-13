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
    def __init__(self, hdf5_path, flow_path, transform=None, num_frames=None, max_frames=None):
        self.hdf5_path = hdf5_path
        self.flow_path = flow_path
        self.transform = transform
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.keys = []
        self.flow_keys = []

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

        with h5py.File(flow_path, "r") as f:
            all_flow_keys = list(f.keys())
            if self.max_frames:
                filtered_flow_keys = []
                for key in all_flow_keys:
                    if f[key]["embeddings"].shape[0] < self.max_frames:
                        filtered_flow_keys.append(key)
                self.flow_keys = filtered_flow_keys
            else:
                self.flow_keys = all_flow_keys

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

        with h5py.File(self.flow_path, "r") as f:
            # note: your flow keys might not match IDs exactly, so adjust as needed
            flow_id = self.keys[idx].split(".")[0]
            group = f[flow_id]
            flow_embeddings = torch.from_numpy(group["embeddings"][:])
            if self.num_frames:
                flow_embeddings = sparse_sampling(flow_embeddings, self.num_frames)
            if self.transform:
                flow_embeddings = self.transform(flow_embeddings)

        return {"video_id": video_id, "embeddings": embeddings.float(), "flow_embeddings": flow_embeddings.float(), "labels": labels, "total_frames": embeddings.shape[0]}


def collate_fn_pad(batch):
    """
    Collate function that pads variable-length sequences for both RGB and Flow,
    and returns distinct boolean masks.
    """
    embeddings = [item["embeddings"] for item in batch]  # RGB
    flow_embeddings = [item["flow_embeddings"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # 1) Pad
    padded_rgb = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)  # (B, T_rgb_max, D)
    padded_flow = torch.nn.utils.rnn.pad_sequence(flow_embeddings, batch_first=True)  # (B, T_flow_max, D)

    # 2) Build mask for each stream
    #   True = real token, False = pad
    lens_rgb = torch.tensor([x.shape[0] for x in embeddings])
    lens_flow = torch.tensor([x.shape[0] for x in flow_embeddings])

    T_rgb_max = padded_rgb.size(1)
    T_flow_max = padded_flow.size(1)

    mask_rgb = torch.arange(T_rgb_max).expand(len(lens_rgb), T_rgb_max)
    mask_rgb = mask_rgb < lens_rgb.unsqueeze(1)
    # shape => (B, T_rgb_max)  (True where it's a real frame)

    mask_flow = torch.arange(T_flow_max).expand(len(lens_flow), T_flow_max)
    mask_flow = mask_flow < lens_flow.unsqueeze(1)
    # shape => (B, T_flow_max)

    return {
        "video_id": [item["video_id"] for item in batch],
        "embeddings": padded_rgb,
        "flow_embeddings": padded_flow,
        "labels": labels,
        "mask_rgb": mask_rgb,  # shape (B, T_rgb_max)
        "mask_flow": mask_flow,  # shape (B, T_flow_max)
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
    flow_dataset = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flow_embeddings_dataset1_si_MLP.h5"

    # Inicializar el dataset y DataLoader
    train_dataset = HDF5VideoDataset(train_dataset, flow_dataset)
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
        print(f"flow_embeddings: {batch['flow_embeddings'].shape}")
        # print(f"labels: {batch['labels']}")
        print(f"lengths: {batch['lengths']}")

    # Llamar a la función para verificar si los datos se cargan correctamente
    check_data_loading(dataloader)
