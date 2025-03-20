import torch
import torch.nn as nn
import clip


class FlowStudentModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device

        # Load CLIP model
        model, preprocess = clip.load(clip_model_name, device=self.device)
        self.visual_encoder = model.visual

    def forward(self, flow_videos):
        """
        Compute embeddings for optical flow frames using CLIP visual encoder.

        Args:
            flow_videos (torch.Tensor): Batch of optical flow videos, shape (B, T, 3, H, W)

        Returns:
            torch.Tensor: Embeddings of shape (B, T, embed_dim)
        """
        B, T, C, H, W = flow_videos.shape

        # Reshape to (B*T, C, H, W) for encoding
        flow_frames = flow_videos.view(B * T, C, H, W)

        # Normalize input images according to CLIP preprocessing
        flow_frames = flow_frames.float() / 255.0  # ensure normalization
        flow_frames = nn.functional.interpolate(flow_frames, size=(224, 224), mode="bilinear")

        # Convert frames to float16 as expected by CLIP
        flow_frames = flow_frames.half()

        # Compute embeddings
        embeddings = self.visual_encoder(flow_frames)

        # Reshape back to (B, T, embed_dim)
        embeddings = embeddings.view(B, T, -1)

        return embeddings


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowStudentModel(clip_model_name="ViT-B/32", device=device).to(device)

    flow_videos = torch.randn(2, 300, 3, 360, 640).to(device)  # Example input
    embeddings = model(flow_videos)  # embeddings shape: (8, 386, embed_dim)
    print(f"Output embeddings shape: {embeddings.shape}")
