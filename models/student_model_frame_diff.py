import torch
import torch.nn as nn
import clip
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


class ResidualMLP(nn.Module):
    def __init__(self, embed_dim, alpha=0.1):
        super().__init__()
        # First linear layer
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        # Non-linear activation (GELU or ReLU could be used; choose as desired)
        self.act = nn.GELU()
        # Second linear layer
        self.fc2 = nn.Linear(embed_dim, embed_dim)

        # A scalar controlling how much of the MLP output to blend back
        # You can make this a learnable parameter by doing:
        #   self.alpha = nn.Parameter(torch.tensor(alpha))
        # or leave it as a fixed float.
        self.alpha = alpha

        # Initialize the second FC layer weights to zero, following some PEFT norms
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: shape (B, T, embed_dim)
        """
        # MLP transform
        mlp_out = self.fc2(self.act(self.fc1(x)))
        # Residual skip connection
        return x + self.alpha * mlp_out


class FrameDiffStudentModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda", num_classes=140, alpha=0.1):
        super().__init__()
        self.device = device

        model, preprocess = clip.load(clip_model_name, device=self.device)
        model = model.float()

        self.preprocess = preprocess
        self.visual_encoder = model.visual
        embed_dim = self.visual_encoder.output_dim

        self.residual_mlp = ResidualMLP(embed_dim, alpha=alpha)

        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, frame_diffs):
        B, T, C, H, W = frame_diffs.shape

        # Reshape to (B*T, C, H, W)
        frame_diffs = frame_diffs.view(B * T, C, H, W).float()

        # Use CLIP's preprocess transformations directly
        clip_preprocess = transforms.Compose(self.preprocess.transforms)
        frame_diffs_processed = torch.stack([clip_preprocess(to_pil_image(frame)) for frame in frame_diffs])

        # Move to correct device and dtype
        frame_diffs_processed = frame_diffs_processed.to(self.device)

        # Compute embeddings
        embeddings = self.visual_encoder(frame_diffs_processed)

        # Reshape back to (B, T, embed_dim)
        embeddings = embeddings.view(B, T, -1)

        # === FROSTER-like residual MLP skip connection ===
        embeddings_for_distillation = self.residual_mlp(embeddings)  # shape (B, T, embed_dim)

        # Mean pooling across temporal dimension
        pooled_embeddings = embeddings.mean(dim=1)

        # Compute logits for classification (float32 precision)
        logits = self.classification_head(pooled_embeddings.float())

        return embeddings, embeddings_for_distillation, logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrameDiffStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=140, alpha=0.1).to(device)

    frame_diff_videos = torch.randint(0, 256, (1, 450, 3, 360, 640), dtype=torch.uint8).to(device)  # Example input
    embeddings, embeddings_for_distillation, logits = model(frame_diff_videos)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings for distillation shape: {embeddings_for_distillation.shape}")
    print(f"Logits shape: {logits.shape}")
