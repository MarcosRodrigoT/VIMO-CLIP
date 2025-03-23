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


class FlowStudentModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda", num_classes=140, alpha=0.1):
        super().__init__()
        self.device = device

        # Load CLIP model
        model, preprocess = clip.load(clip_model_name, device=self.device)
        model = model.float()  # put CLIP in float32

        self.preprocess = preprocess
        self.visual_encoder = model.visual
        embed_dim = self.visual_encoder.output_dim

        # A small 2-layer MLP with skip connection, as in the FROSTER paper figure 3.
        self.residual_mlp = ResidualMLP(embed_dim, alpha=alpha)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, flow_videos):
        """
        Compute embeddings and classification logits for optical flow frames using CLIP visual encoder.

        Args:
            flow_videos (torch.Tensor): (B, T, 3, H, W)

        Returns:
            tuple: embeddings (B, T, embed_dim), logits (B, num_classes)
        """
        B, T, C, H, W = flow_videos.shape

        # Reshape to (B*T, C, H, W)
        flow_frames = flow_videos.view(B * T, C, H, W).float()

        # Use CLIP's preprocess transformations directly
        clip_preprocess = transforms.Compose(self.preprocess.transforms)
        flow_frames_processed = torch.stack([clip_preprocess(to_pil_image(frame)) for frame in flow_frames])

        # Move to correct device and dtype
        flow_frames_processed = flow_frames_processed.to(self.device)  # .half() in case we are working with CLIP in float16

        # Compute embeddings
        embeddings = self.visual_encoder(flow_frames_processed)

        # Reshape back to (B, T, embed_dim)
        embeddings = embeddings.view(B, T, -1)

        # === FROSTER-like residual MLP skip connection ===
        embeddings_for_distillation = self.residual_mlp(embeddings)  # shape (B, T, embed_dim)

        # Mean pooling across temporal dimension
        pooled_embeddings = embeddings_for_distillation.mean(dim=1)

        # Compute logits for classification (float32 precision)
        logits = self.classification_head(pooled_embeddings.float())

        return embeddings, embeddings_for_distillation, logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=140, alpha=0.1).to(device)

    flow_videos = torch.randint(0, 256, (1, 450, 3, 360, 640), dtype=torch.uint8).to(device)  # Example input
    embeddings, embeddings_for_distillation, logits = model(flow_videos)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings for distillation shape: {embeddings_for_distillation.shape}")
    print(f"Logits shape: {logits.shape}")
