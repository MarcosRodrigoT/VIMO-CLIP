import torch
import torch.nn.functional as F


def distillation_loss(student_embeddings, teacher_embeddings, mode="mse"):
    """
    Compute the distillation loss between student and teacher embeddings.

    Args:
        student_embeddings (torch.Tensor): Embeddings from the student model. Shape: (B, T, embed_dim)
        teacher_embeddings (torch.Tensor): Embeddings from the teacher model. Shape: (B, T, embed_dim)
        mode (str): 'mse' for mean squared error or 'cosine' for cosine similarity loss.

    Returns:
        torch.Tensor: Calculated distillation loss (scalar).
    """
    if mode == "mse":
        loss = F.mse_loss(student_embeddings, teacher_embeddings)
    elif mode == "cosine":
        # This is more clean but produces NaNs
        # loss = 1 - F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1).mean()

        # Manual computation due to NaNs appearing because of a division by 0
        epsilon = 1e-5

        # Compute norms and clamp to epsilon
        student_norm = student_embeddings.norm(dim=-1).clamp(min=epsilon)
        teacher_norm = teacher_embeddings.norm(dim=-1).clamp(min=epsilon)

        # Compute cosine similarity safely
        cosine_sim = (student_embeddings * teacher_embeddings).sum(dim=-1) / (student_norm * teacher_norm)

        # Clamp cosine_sim between -1 and 1 to avoid numerical instability
        cosine_sim = cosine_sim.clamp(-1 + epsilon, 1 - epsilon)

        # Compute loss
        loss = 1 - cosine_sim.mean()
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'mse' or 'cosine'.")

    return loss


def classification_loss(predictions, targets):
    """
    Compute the binary cross-entropy loss for multi-label classification.

    Args:
        predictions (torch.Tensor): Predictions from the model (logits). Shape: (B, num_classes)
        targets (torch.Tensor): Ground truth labels. Shape: (B, num_classes)

    Returns:
        torch.Tensor: Calculated binary cross-entropy loss (scalar).
    """
    return F.binary_cross_entropy_with_logits(predictions, targets.float())


def reconstruction_loss(reconstruction, input):
    """
    Compute the reconstruction loss between the reconstruction and the input.

    Args:
        reconstruction (torch.Tensor):
        input (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Example distillation loss
    student_emb = torch.randn(8, 10, 512)  # Example shape: (batch_size, sequence_length, embed_dim)
    teacher_emb = torch.randn(8, 10, 512)  # Example shape: (batch_size, sequence_length, embed_dim)

    dist_loss = distillation_loss(student_emb, teacher_emb, mode="cosine")  # or mode='mse'
    print(f"Distillation loss: {dist_loss}")

    # Example classification loss
    pred_logits = torch.randn(8, 140)  # Example logits output from your model
    target_labels = torch.randint(0, 2, (8, 140)).float()  # Example ground truth labels

    class_loss = classification_loss(pred_logits, target_labels)
    print(f"Classification loss: {class_loss}")
