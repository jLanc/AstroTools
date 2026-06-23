"""
AstroDeNoise — U-Net model for noise reduction in linear astrophotography.

Architecture: U-Net with residual blocks
- Encoder captures multi-scale context (broad sky background → fine star detail)
- Skip connections preserve spatial information at every scale
- Decoder reconstructs the clean image at full resolution
- Residual output: model predicts the *noise to subtract*, not the full image
  (this is critical — in linear space we want to remove noise while leaving
  real signal completely untouched)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-conv residual block with GroupNorm and optional dropout.
    
    Dropout is important for small datasets — it prevents the model from
    memorising training patches and forces it to learn generalisable signal
    features instead of per-image noise patterns.
    
    Rule of thumb for dropout rate:
      <50 subs:   0.2–0.3  (strong regularisation)
      50–200 subs: 0.1
      200+ subs:  0.05 or 0.0
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class EncoderBlock(nn.Module):
    """Downsample + residual processing."""
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2, dropout: float = 0.1):
        super().__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.res_blocks = nn.Sequential(*[ResidualBlock(out_ch, dropout) for _ in range(num_res)])

    def forward(self, x):
        x = self.downsample(x)
        return self.res_blocks(x)


class DecoderBlock(nn.Module):
    """Upsample + skip connection merge + residual processing."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_res: int = 2, dropout: float = 0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.merge_conv = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.res_blocks = nn.Sequential(*[ResidualBlock(out_ch, dropout) for _ in range(num_res)])

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle odd-dimension edge cases
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.merge_conv(x)
        return self.res_blocks(x)


class AstroDeNoiseNet(nn.Module):
    """
    U-Net for noise reduction in linear astrophotography.

    Input:  (B, C, H, W) — linear normalised image, float32 [0, 1]
    Output: (B, C, H, W) — denoised linear image, float32 [0, 1]

    Args:
        in_channels:   1 for mono, 3 for OSC colour
        base_features: feature width at first encoder stage (doubles each level)
        depth:         encoder/decoder levels — 4 gives 16× spatial compression
        dropout:       spatial dropout rate in residual blocks

    Scaling guide — choose based on your total number of subs across ALL targets:
        < 50 subs   : base_features=24, dropout=0.25  →  ~9M params  (prevents overfitting)
        50–200 subs : base_features=32, dropout=0.10  → ~27M params  (default)
        200–500 subs: base_features=48, dropout=0.05  → ~60M params
        500+ subs   : base_features=64, dropout=0.0   → ~108M params

    More parameters with too little data = the model memorises noise patterns
    rather than learning to remove them. Let build_model() handle this via --num_subs.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, base_features), num_channels=base_features),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_features, dropout),
        )

        ch = base_features
        self.encoders = nn.ModuleList()
        self.enc_channels = [base_features]
        for _ in range(depth):
            out_ch = min(ch * 2, 512)
            self.encoders.append(EncoderBlock(ch, out_ch, dropout=dropout))
            self.enc_channels.append(out_ch)
            ch = out_ch

        self.bottleneck = nn.Sequential(
            ResidualBlock(ch, dropout),
            ResidualBlock(ch, dropout),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth):
            skip_ch = self.enc_channels[depth - i - 1]
            out_ch  = self.enc_channels[depth - i - 1]
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch, dropout=dropout))
            ch = out_ch

        self.head = nn.Sequential(
            ResidualBlock(ch, dropout=0.0),   # no dropout at output — stability
            nn.Conv2d(ch, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        features = self.stem(x)

        # Encode — save skip connections
        skips = [features]
        enc = features
        for encoder in self.encoders:
            enc = encoder(enc)
            skips.append(enc)

        # Bottleneck
        enc = self.bottleneck(enc)

        # Decode — use skip connections in reverse order
        dec = enc
        for i, decoder in enumerate(self.decoders):
            skip = skips[self.depth - i - 1]
            dec = decoder(dec, skip)

        # Residual output: the head predicts noise to remove.
        # Adding a negative residual to x subtracts noise while leaving
        # signal untouched. Clamp to [0, 1] to stay in linear range.
        residual = self.head(dec)
        return torch.clamp(x + residual, 0.0, 1.0)


def build_model(in_channels: int = 3, device: str = 'cpu', num_subs: int = 0) -> AstroDeNoiseNet:
    """
    Factory: build model scaled to your dataset size.

    Pass num_subs = total number of subs across all training targets to get
    automatically tuned base_features and dropout. Or set num_subs=0 to use
    the default (32 features, 0.1 dropout).
    """
    if num_subs >= 500:
        base_features, dropout = 64, 0.00
    elif num_subs >= 200:
        base_features, dropout = 48, 0.05
    elif num_subs >= 50:
        base_features, dropout = 32, 0.10
    elif num_subs > 0:
        base_features, dropout = 24, 0.25   # small dataset — strong regularisation
    else:
        base_features, dropout = 32, 0.10   # default

    model = AstroDeNoiseNet(
        in_channels=in_channels,
        base_features=base_features,
        depth=4,
        dropout=dropout,
    )
    print(f"Model: base_features={base_features}, dropout={dropout}, "
          f"params={count_parameters(model):,}")
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = build_model(in_channels=3)
    print(f"AstroDeNoiseNet — {count_parameters(model):,} trainable parameters")
    dummy = torch.zeros(1, 3, 256, 256)
    out = model(dummy)
    print(f"Input shape: {dummy.shape}  →  Output shape: {out.shape}")
