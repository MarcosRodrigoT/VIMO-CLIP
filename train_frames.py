import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset_frames import HDF5VideoDataset, collate_fn
from models.student_model import FlowStudentModel
from losses import distillation_loss, classification_loss
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def evaluate(model, val_loader, device, distillation_loss_mode, class_positive_weight):
    model.eval()
    val_bar = tqdm(val_loader, desc="Validation", leave=False)

    epoch_distill_loss, epoch_class_loss, epoch_total_loss = 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_bar:
            embeddings_gt = batch["embeddings"].to(device)
            flow_videos = batch["flow_video"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            student_emb, student_emb_for_distill, logits = model(flow_videos)

            distill_loss = distillation_loss(student_emb_for_distill, embeddings_gt[:, :-1, :], mode=distillation_loss_mode)
            class_loss = classification_loss(logits, labels, positive_weight=class_positive_weight)
            total_loss = distill_loss + class_loss

            epoch_distill_loss += distill_loss.item()
            epoch_class_loss += class_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            val_bar.set_postfix({"ValDist": f"{distill_loss.item():.4f}", "ValClass": f"{class_loss.item():.4f}", "ValTotal": f"{total_loss.item():.4f}"})

    if num_batches == 0:
        # Edge case: if the val set is empty or all are mismatched => avoid division by zero
        return 0.0, 0.0, 0.0

    distill_loss_avg = epoch_distill_loss / num_batches
    class_loss_avg = epoch_class_loss / num_batches
    total_loss_avg = epoch_total_loss / num_batches

    return distill_loss_avg, class_loss_avg, total_loss_avg


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 10
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-3
    distillation_loss_mode = "cosine"
    num_classes = 140
    grad_clip_norm = None
    sequence_length = 30
    residual_alpha = 0.1
    class_positive_weight = 9

    # === Dataset paths ===
    train_hdf5_path = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_train_clip_vit32.h5"
    val_hdf5_path = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/ak_val_clip_vit32.h5"
    flow_videos_dir = "/mnt/Data/enz/AnimalKingdom/action_recognition/dataset/flows"

    # === Dataset and DataLoader ===
    train_dataset = HDF5VideoDataset(clip_embeddings_dir=train_hdf5_path, flow_videos_dir=flow_videos_dir, sequence_length=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)

    val_dataset = HDF5VideoDataset(clip_embeddings_dir=val_hdf5_path, flow_videos_dir=flow_videos_dir, sequence_length=sequence_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    # === Model, optimizer ===
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=num_classes, alpha=residual_alpha).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # === Logging & checkpoint setup ===
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")  # e.g., "20231105-183122"
    log_dir = f"logs/{run_name}"
    ckpt_dir = f"checkpoints/{run_name}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    best_val_loss = float("inf")  # track best val total loss

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        epoch_distill_loss, epoch_class_loss, epoch_total_loss = 0.0, 0.0, 0.0

        # Training progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")

        for batch in train_bar:
            embeddings_gt = batch["rgb_emb"].to(device)  # shape (B, T, embed_dim)
            flow_frames = batch["flow_frames"].to(device)  # shape (B, T-1, 3, H, W)
            labels = batch["labels"].to(device)  # shape (B, num_classes)

            # Student forward
            student_embeddings, student_embeddings_for_distillation, logits = model(flow_frames)  # (B, T-1, embed_dim), (B, T-1, embed_dim), (B, num_classes)

            # Compute losses
            # TODO: Changed distillation loss: From student_embeddings_for_distillation / teacher_emb_diff to student_embeddings_for_distillation / embeddings_gt[:, :-1, :]
            #  We use embeddings_gt[:, :-1, :] so that it has the same shape as student_embeddings_for_distillation (we renive the last T frame)
            distill_loss = distillation_loss(student_embeddings_for_distillation, embeddings_gt[:, :-1, :], mode=distillation_loss_mode)
            class_loss = classification_loss(logits, labels, positive_weight=class_positive_weight)

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

            # Write step-wise losses to TensorBoard
            writer.add_scalar("Loss/Distillation", distill_loss.item(), global_step)
            writer.add_scalar("Loss/Classification", class_loss.item(), global_step)
            writer.add_scalar("Loss/Total", total_loss.item(), global_step)

            train_bar.set_postfix({"Distill Loss": f"{distill_loss.item():.4f}", "Class Loss": f"{class_loss.item():.4f}", "Total Loss": f"{total_loss.item():.4f}"})

            global_step += 1

        # === Averages for training this epoch ===
        num_batches = len(train_loader)
        avg_train_dist = epoch_distill_loss / num_batches
        avg_train_class = epoch_class_loss / num_batches
        avg_train_total = epoch_total_loss / num_batches

        # === Validation ===
        val_dist, val_class, val_total = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            distillation_loss_mode=distillation_loss_mode,
            class_positive_weight=class_positive_weight,
        )

        # === Logging epoch-level results ===
        writer.add_scalar("Train/EpochLoss/Distillation", avg_train_dist, epoch)
        writer.add_scalar("Train/EpochLoss/Classification", avg_train_class, epoch)
        writer.add_scalar("Train/EpochLoss/Total", avg_train_total, epoch)

        writer.add_scalar("Val/EpochLoss/Distillation", val_dist, epoch)
        writer.add_scalar("Val/EpochLoss/Classification", val_class, epoch)
        writer.add_scalar("Val/EpochLoss/Total", val_total, epoch)

        # Log last batch’s logits & labels for this epoch
        logits_text = str(logits)
        labels_text = str(labels)
        writer.add_text("Logits/LastBatch", logits_text, epoch)
        writer.add_text("Labels/LastBatch", labels_text, epoch)
        writer.add_histogram("Logits/LastBatch", logits, epoch)
        writer.add_histogram("Labels/LastBatch", labels, epoch)

        # Print epoch results
        print(f"\nEpoch [{epoch + 1}/{epochs}] completed:")
        print("Training results:")
        print(f"- Avg Distillation Loss: {avg_train_dist:.4f}")
        print(f"- Avg Classification Loss: {avg_train_class:.4f}")
        print(f"- Avg Total Loss: {avg_train_total:.4f}\n")
        print("Validation results:")
        print(f"- Avg Distillation Loss: {val_dist:.4f}")
        print(f"- Avg Classification Loss: {val_class:.4f}")
        print(f"- Avg Total Loss: {val_total:.4f}\n")

        # Save checkpoint
        torch.save(model.state_dict(), f"{ckpt_dir}/student_epoch_{epoch+1}.pth")

        # Check if this is the best so far => save to "<run_name> - best"
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_ckpt_dir = f"{ckpt_dir} - best"
            os.makedirs(best_ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{best_ckpt_dir}/student_best.pth")
            print(f"=> New best val loss: {best_val_loss:.4f}. Saved to {best_ckpt_dir}/student_best.pth")


if __name__ == "__main__":
    train()
