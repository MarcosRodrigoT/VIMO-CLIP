import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import HDF5VideoDataset, collate_fn
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
            embeddings_gt = batch["rgb_emb"].to(device)
            flow_videos = batch["flow_frames"].to(device)
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


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Dataset and DataLoader ===
    train_dataset = HDF5VideoDataset(clip_embeddings_dir=args.train_hdf5_path, flow_videos_dir=args.flow_videos_dir, sequence_length=args.sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    val_dataset = HDF5VideoDataset(clip_embeddings_dir=args.val_hdf5_path, flow_videos_dir=args.flow_videos_dir, sequence_length=args.sequence_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # === Model, optimizer ===
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=args.num_classes, alpha=args.residual_alpha).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

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
    for epoch in range(args.epochs):
        model.train()
        epoch_distill_loss, epoch_class_loss, epoch_total_loss = 0.0, 0.0, 0.0

        # Training progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)")

        for batch in train_bar:
            embeddings_gt = batch["rgb_emb"].to(device)  # shape (B, T, embed_dim)
            flow_frames = batch["flow_frames"].to(device)  # shape (B, T-1, 3, H, W)
            labels = batch["labels"].to(device)  # shape (B, num_classes)

            # Student forward
            student_embeddings, student_embeddings_for_distillation, logits = model(flow_frames)  # (B, T-1, embed_dim), (B, T-1, embed_dim), (B, num_classes)

            # Compute losses
            distill_loss = distillation_loss(student_embeddings_for_distillation, embeddings_gt[:, :-1, :], mode=args.distillation_loss_mode)
            class_loss = classification_loss(logits, labels, positive_weight=args.class_positive_weight)
            total_loss = distill_loss + class_loss

            # Backward and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
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
            distillation_loss_mode=args.distillation_loss_mode,
            class_positive_weight=args.class_positive_weight,
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
        print(f"\nEpoch [{epoch + 1}/{args.epochs}] completed:")
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
    parser = argparse.ArgumentParser(description="Train flow-only student model")

    # Core training hyper-parameters
    parser.add_argument("--epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker processes used by the DataLoader.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate for the optimizer.")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Clip gradient L2-norm to this value (disabled if None).")

    # Model / loss settings
    parser.add_argument("--distillation-loss-mode", type=str, default="cosine", choices=["cosine", "mse"], help="Similarity metric used for the distillation loss.")
    parser.add_argument("--num-classes", type=int, default=140, help="Number of action classes in the dataset.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Number of frames per video clip passed to the model.")
    parser.add_argument("--residual-alpha", type=float, default=0.1, help="Scaling factor for the residual connection in the student model.")
    parser.add_argument("--class-positive-weight", type=float, default=9, help="Positive-class weight for the BCE classification loss.")

    # Dataset paths
    parser.add_argument(
        "--train-hdf5-path",
        type=str,
        default="dataset/embeddings/train_clip_embeddings.h5",
        help="Path to the HDF5 file with CLIP embeddings for the training split.",
    )
    parser.add_argument(
        "--val-hdf5-path",
        type=str,
        default="dataset/embeddings/val_clip_embeddings.h5",
        help="Path to the HDF5 file with CLIP embeddings for the validation split.",
    )
    parser.add_argument(
        "--flow-videos-dir",
        type=str,
        default="dataset/flows",
        help="Directory containing optical-flow frame folders.",
    )

    args = parser.parse_args()
    train(args)
