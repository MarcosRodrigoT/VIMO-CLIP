import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"

        # Capas de atención
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Feed-Forward con activación GELU/ReLU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Normalizaciones y dropouts
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross_src=None, src_key_padding_mask=None, cross_key_padding_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm_self(x + self.dropout(attn_output))

        # Cross-attention
        if cross_src is not None:
            attn_output, _ = self.cross_attn(x, cross_src, cross_src, key_padding_mask=cross_key_padding_mask)
            x = self.norm_cross(x + self.dropout(attn_output))

        # Feed-Forward
        ffn_output = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ffn_output))

        return x


class AMO_CLIP_WO_MASK(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        num_classes=140,
        use_cross_attention=True,
        use_pe=False,
        use_only_rgb=False,
        use_only_flow=False,
        concat_dim=1,
        dropout=0.1,
        mlp_dropout=0.3,
        device="cuda",
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention
        self.use_pe = use_pe
        self.use_only_rgb = use_only_rgb
        self.use_only_flow = use_only_flow
        self.concat_dim = concat_dim
        self.d_model = d_model
        self.device = device

        self.layers = nn.ModuleList([AttentionLayer(d_model, nhead, dim_feedforward, dropout=dropout) for _ in range(num_layers)])

        # MLP classifier
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(mlp_dropout), nn.Linear(d_model // 2, num_classes))

        self.projection_layer = nn.Linear(2 * self.d_model, self.d_model)

    def positional_encoding(self, seq_len):
        """
        Genera un positional encoding sinusoidal para una secuencia de longitud seq_len.
        """
        position = torch.arange(seq_len).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).to(self.device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model).to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, rgb_emb, motion_emb, mask_rgb=None, mask_flow=None):
        """
        video_input: [B, T, 512]
        RGB embeddings of the video frames
        Motion augmented embeddings of the video frames

        A: concatenate the RGB and Motion augmented embeddings as input to the transformer encoder
            1. conatenate the RGB and Motion augmented embeddings
            2. pass the concatenated embeddings through the transformer encoder
            3. pass the transformer encoder output through the classifier


        B: pass the RGB embeddings as the input to the transformer encoder and attend the motion augmented embeddings
        through cross-attention
            1. pass the RGB embeddings through the transformer encoder
            2. attend the motion augmented embeddings through cross-attention
            3. pass the attended embeddings through the classifier

        """
        # If a mask is provided with shape [B, T], we invert it for key_padding_mask
        # because MultiheadAttention expects True for "pad" tokens to ignore.
        # The tilde (~) operator is bitwise NOT, which flips booleans:
        #   True -> False, False -> True.
        # So if your mask is "True means real data, False means pad",
        # you do: attn_mask = ~mask
        # multiheadattention expects True for "ignore," so invert them:
        attn_rgb = ~mask_rgb if mask_rgb is not None else None
        attn_flow = ~mask_flow if mask_flow is not None else None

        # positional encoding
        if self.use_pe:
            pe_rgb = self.positional_encoding(rgb_emb.size(1)).to(rgb_emb.device)  # [T_rgb, 512]
            pe_flow = self.positional_encoding(motion_emb.size(1)).to(motion_emb.device)  # [T_flow, 512]
            # Añadir positional encoding a los embeddings
            rgb_emb += pe_rgb.unsqueeze(0)  # [B, T_rgb, 512]
            motion_emb += pe_flow.unsqueeze(0)  # [B, T_flow, 512]

        if self.use_only_rgb:
            # Si solo se utilizan los embeddings RGB
            x = rgb_emb
            for layer in self.layers:
                x = layer(x)
        elif self.use_only_flow:
            # Si solo se utilizan los embeddings de flujo
            x = motion_emb
            for layer in self.layers:
                x = layer(x)
        elif self.use_cross_attention:
            # Si se utilizan embeddings de OF y RGB + Cross-Attention
            x = rgb_emb
            for layer in self.layers:
                x = layer(x, cross_src=motion_emb)
        elif not self.use_cross_attention:
            # Si se utilizan embeddings de OF y RGB + Self-Attention
            if self.concat_dim == 1:
                # Concatenación TEMPORAL
                x = torch.cat([rgb_emb, motion_emb], dim=self.concat_dim)
            elif self.concat_dim == -1:
                # Concatenación EMBEDDINGS
                rgb_emb = rgb_emb[:, :-1, :]
                x = torch.cat([rgb_emb, motion_emb], dim=self.concat_dim)
                x = self.projection_layer(x)

            # Aplicar self-attention a secuencia extendida
            for layer in self.layers:
                x = layer(x)

        # Pooling sobre TODA la secuencia (original + extendida)
        logits = self.classifier(x.mean(dim=1))
        return logits
