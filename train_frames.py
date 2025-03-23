import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset_frames import HDF5VideoDataset, collate_fn
from models.student_model import FlowStudentModel
from losses import distillation_loss, classification_loss
import os
from tqdm import tqdm


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 30
    batch_size = 32
    num_workers = 4
    learning_rate = 3e-4
    distillation_loss_mode = "cosine"
    num_classes = 140
    grad_clip_norm = None
    sequence_length = 30
    residual_alpha = 0.1

    # === Dataset paths ===
    train_hdf5_path = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_train_clip_vit32.h5"
    flow_videos_dir = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"

    # === Dataset and DataLoader ===
    train_dataset = HDF5VideoDataset(clip_embeddings_dir=train_hdf5_path, flow_videos_dir=flow_videos_dir, sequence_length=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)

    # === Model, optimizer ===
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=num_classes, alpha=residual_alpha).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        epoch_distill_loss, epoch_class_loss, epoch_total_loss = 0.0, 0.0, 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            embeddings_gt = batch["rgb_emb"].to(device)  # shape (B, T, embed_dim)
            flow_frames = batch["flow_frames"].to(device)  # shape (B, T-1, 3, H, W)
            labels = batch["labels"].to(device)  # shape (B, num_classes)

            # Compute embedding differences (teacher)
            teacher_emb_diff = embeddings_gt[:, 1:, :] - embeddings_gt[:, :-1, :]  # shape (B, T-1, embed_dim)

            # Student forward
            student_embeddings, student_embeddings_for_distillation, logits = model(flow_frames)  # (B, T-1, embed_dim), (B, T-1, embed_dim), (B, num_classes)

            # Compute losses
            distill_loss = distillation_loss(student_embeddings_for_distillation, teacher_emb_diff, mode=distillation_loss_mode)
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

        # Logging after each epoch
        num_batches = len(train_loader)
        avg_distill = epoch_distill_loss / num_batches
        avg_class = epoch_class_loss / num_batches
        avg_total = epoch_total_loss / num_batches

        print(f"\nEpoch [{epoch+1}/{epochs}] completed:")
        print(f"Avg Distillation Loss: {avg_distill:.4f}")
        print(f"Avg Classification Loss: {avg_class:.4f}")
        print(f"Avg Total Loss: {avg_total:.4f}\n")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/student_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
