import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import HDF5VideoDataset, collate_fn_pad
from models.student_model import FlowStudentModel
from losses import distillation_loss, classification_loss
import os
from tqdm import tqdm


def compute_embedding_differences(embeddings):
    """
    Compute differences between consecutive embeddings.
    embeddings: shape (B, T, embed_dim)
    returns: shape (B, T-1, embed_dim)
    """
    return embeddings[:, 1:, :] - embeddings[:, :-1, :]


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Hyperparameters ===
    epochs = 10
    batch_size = 1
    num_workers = 1
    learning_rate = 1e-4
    distillation_loss_mode = "cosine"
    num_classes = 140
    grad_clip_norm = None

    # === Dataset paths ===
    train_hdf5_path = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_train_clip_vit32.h5"
    flow_videos_dir = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"

    # === Dataset and DataLoader ===
    train_dataset = HDF5VideoDataset(clip_embeddings_dir=train_hdf5_path, flow_videos_dir=flow_videos_dir, num_frames=None, max_frames=500)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad, num_workers=num_workers)

    # === Model, optimizer ===
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=num_classes).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        epoch_distill_loss, epoch_class_loss, epoch_total_loss = 0.0, 0.0, 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            embeddings_gt = batch["embeddings"].to(device)  # Teacher embeddings
            flow_videos = batch["flow_video"].to(device)  # Optical flow videos
            labels = batch["labels"].to(device)  # Labels for classification

            # Compute embedding differences (teacher)
            teacher_emb_diff = compute_embedding_differences(embeddings_gt)  # (B, T-1, embed_dim)

            # Student model forward
            student_embeddings, logits = model(flow_videos)  # (B, T-1, embed_dim), (B, num_classes)

            # Compute losses
            # TODO: Meter MLP ente medias de student_embeddings y teacher_emb_diff como hacen en FROSTER
            distill_loss = distillation_loss(student_embeddings, teacher_emb_diff, mode=distillation_loss_mode)
            class_loss = classification_loss(logits, labels)

            # TODO: Maybe add a balance factor
            total_loss = distill_loss + class_loss

            # Backward and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            # Accumulate losses for logging
            epoch_distill_loss += distill_loss.item()
            epoch_class_loss += class_loss.item()
            epoch_total_loss += total_loss.item()

            progress_bar.set_postfix({"Distill Loss": f"{distill_loss.item():.4f}", "Class Loss": f"{class_loss.item():.4f}", "Total Loss": f"{total_loss.item():.4f}"})

        num_batches = len(train_loader)
        avg_distill_loss = epoch_distill_loss / num_batches
        avg_class_loss = epoch_class_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        print(f"\nEpoch [{epoch + 1}/{epochs}] completed:")
        print(f"Avg Distillation Loss: {avg_distill_loss:.4f}")
        print(f"Avg Classification Loss: {avg_class_loss:.4f}")
        print(f"Avg Total Loss: {avg_total_loss:.4f}\n")

        # Save checkpoint every epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/student_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
