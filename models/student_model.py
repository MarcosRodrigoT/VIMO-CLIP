import torch
import torch.nn as nn
import clip
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


class FlowStudentModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda", num_classes=140):
        super().__init__()
        self.device = device

        # Load CLIP model
        model, preprocess = clip.load(clip_model_name, device=self.device)
        self.preprocess = preprocess
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
            flow_videos (torch.Tensor): (B, T, 3, H, W), values expected in [0, 255]

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
        flow_frames_processed = flow_frames_processed.to(self.device).half()

        # Compute embeddings
        embeddings = self.visual_encoder(flow_frames_processed)

        # Reshape back to (B, T, embed_dim)
        embeddings = embeddings.view(B, T, -1)

        # Mean pooling across temporal dimension
        pooled_embeddings = embeddings.mean(dim=1)

        # Compute logits for classification (float32 precision)
        logits = self.classification_head(pooled_embeddings.float())

        return embeddings, logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device, num_classes=140).to(device)

    flow_videos = torch.randint(0, 256, (2, 300, 3, 360, 640), dtype=torch.uint8).to(device)  # Example input
    embeddings, logits = model(flow_videos)
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output logits shape: {logits.shape}")
