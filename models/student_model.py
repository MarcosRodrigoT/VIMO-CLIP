import torch
import torch.nn as nn
import clip


class FlowStudentModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda", num_classes=140):
        super().__init__()
        self.device = device

        # Load CLIP model
        model, preprocess = clip.load(clip_model_name, device=self.device)
        self.visual_encoder = model.visual
        embed_dim = self.visual_encoder.output_dim

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
            flow_videos (torch.Tensor): Batch of optical flow videos, shape (B, T, 3, H, W)

        Returns:
            tuple: embeddings shape (B, T, embed_dim), logits shape (B, num_classes)
        """
        B, T, C, H, W = flow_videos.shape

        # Reshape to (B*T, C, H, W) for encoding
        flow_frames = flow_videos.view(B * T, C, H, W)

        # Normalize input images and resize according to CLIP preprocessing
        flow_frames = flow_frames.float() / 255.0
        flow_frames = nn.functional.interpolate(flow_frames, size=(224, 224), mode="bilinear")

        # Convert frames to float16 as expected by CLIP
        flow_frames = flow_frames.half()

        # Compute embeddings
        embeddings = self.visual_encoder(flow_frames)

        # Reshape back to (B, T, embed_dim)
        embeddings = embeddings.view(B, T, -1)

        # Pool embeddings across temporal dimension (mean pooling)
        pooled_embeddings = embeddings.mean(dim=1)

        # Compute logits for classification
        logits = self.classification_head(pooled_embeddings.float())

        return embeddings, logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=140).to(device)

    flow_videos = torch.randn(2, 300, 3, 360, 640).to(device)  # Example input
    embeddings, logits = model(flow_videos)
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output logits shape: {logits.shape}")
