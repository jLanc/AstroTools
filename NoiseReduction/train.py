"""
AstroDeNoise Training Script — Noise Reduction

Usage:
    python train.py --data_root ./data --epochs 100 --channels 3

Recommended workflow:
    1. Organise your data (see dataset.py for structure)
    2. Run train.py — it checkpoints every N epochs and saves the best model
    3. Monitor the loss curves printed to console
    4. Run infer.py on new subs or an already-stacked master frame using the saved model

Hardware:
    - NVIDIA GPU strongly recommended (CUDA). A 4GB GPU can handle patch_size=128.
    - 8GB+ GPU: patch_size=256, batch_size=4
    - CPU-only: works but training will take hours per epoch — reduce dataset size
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from model import build_model
from dataset import AstroDataset, AstroDatasetPreloaded


# ── Loss Functions ───────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Build a normalised 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.outer(g)                                    # (W, W)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)          # (1, 1, W, W)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss.

    SSIM measures three things simultaneously:
      - Luminance:  are the local mean brightnesses the same?
      - Contrast:   are the local standard deviations the same?
      - Structure:  are the local spatial patterns correlated?

    For noise reduction this is much better than pure MAE because:
    - A noisy output and a clean output can have similar MAE against the target
      (noise is zero-mean, so errors cancel) yet look completely different visually.
    - SSIM specifically penalises variance in the prediction that isn't in the target
      — which is exactly what noise is.

    Returns 1 - SSIM so that lower = better, consistent with other loss terms.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels    = channels
        self.C1 = 0.01 ** 2   # stability constants (standard values)
        self.C2 = 0.03 ** 2
        kernel = _gaussian_kernel(window_size, sigma, channels)
        self.register_buffer('kernel', kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        k   = self.kernel

        # Local means
        mu_p  = F.conv2d(pred,   k, padding=pad, groups=self.channels)
        mu_t  = F.conv2d(target, k, padding=pad, groups=self.channels)
        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        # Local variances and covariance
        sigma_p2  = F.conv2d(pred * pred,     k, padding=pad, groups=self.channels) - mu_p2
        sigma_t2  = F.conv2d(target * target, k, padding=pad, groups=self.channels) - mu_t2
        sigma_pt  = F.conv2d(pred * target,   k, padding=pad, groups=self.channels) - mu_pt

        # SSIM map
        num   = (2 * mu_pt  + self.C1) * (2 * sigma_pt  + self.C2)
        denom = (mu_p2 + mu_t2 + self.C1) * (sigma_p2 + sigma_t2 + self.C2)
        ssim_map = num / (denom + 1e-8)

        return 1.0 - ssim_map.mean()


class NoiseSuppressionLoss(nn.Module):
    """
    Penalises variance in the model output specifically in regions where
    the target (stack) is smooth.

    How it works:
      1. Compute local variance of the target in a small neighbourhood.
      2. Where target local variance is low → that region is smooth sky / clean background.
      3. In those regions, penalise any local variance in the prediction.
         Because if the target is smooth there and the model output is rough,
         the model is either leaving noise in or hallucinating texture.

    This directly targets the most common noise reduction failure mode:
    noise remaining in flat background regions while structure is preserved.
    """
    def __init__(self, kernel_size: int = 7, smooth_threshold: float = 5e-4):
        super().__init__()
        self.k   = kernel_size
        self.thr = smooth_threshold
        pad      = kernel_size // 2
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=pad)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Local variance of target: Var(X) = E[X²] - E[X]²
        tgt_mean  = self.pool(target)
        tgt_var   = self.pool(target ** 2) - tgt_mean ** 2

        # Smooth mask: regions where target variance is below threshold
        # Average across channels so a single spatial mask applies to all
        smooth_mask = (tgt_var.mean(dim=1, keepdim=True) < self.thr).float()

        # Local variance of prediction in those same regions
        pred_mean = self.pool(pred)
        pred_var  = self.pool(pred ** 2) - pred_mean ** 2

        # Only penalise prediction variance where the target is smooth
        loss = (pred_var * smooth_mask).mean()
        return loss


class AstroNoiseLoss(nn.Module):
    """
    Combined loss for astrophotography noise reduction.

    Components and why each one is here:

    1. MAE (L1):               Pixel-accurate reconstruction. Keeps the model
                               honest about brightness values.

    2. SSIM:                   Structural similarity. Explicitly penalises noise
                               (variance present in prediction but not in target)
                               while rewarding preservation of real edges and
                               structure. The single most important term for
                               perceptual noise reduction quality.

    3. Gradient loss:          Ensures real edges (star edges, nebula/sky boundaries)
                               are preserved at full sharpness. Prevents the model
                               from softening real detail to achieve a lower SSIM.

    4. Noise suppression loss: Directly targets noise remaining in smooth background
                               regions — the most visible noise reduction failure.

    NOT included (vs the original loss):
    - Log-space MAE: that term over-weighted faint pixel errors to boost faint signal
      recovery. For noise reduction it would bias the model toward over-smoothing
      the sky background at the expense of preserving faint real structure.
    """
    def __init__(
        self,
        mae_weight:   float = 1.0,
        ssim_weight:  float = 1.0,
        grad_weight:  float = 0.4,
        noise_weight: float = 0.3,
        channels:     int   = 3,
    ):
        super().__init__()
        self.mae_w   = mae_weight
        self.ssim_w  = ssim_weight
        self.grad_w  = grad_weight
        self.noise_w = noise_weight

        self.ssim  = SSIMLoss(channels=channels)
        self.noise = NoiseSuppressionLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mae        = F.l1_loss(pred, target)
        ssim_loss  = self.ssim(pred, target)
        grad_loss  = self._gradient_loss(pred, target)
        noise_loss = self.noise(pred, target)

        total = (
            self.mae_w   * mae       +
            self.ssim_w  * ssim_loss +
            self.grad_w  * grad_loss +
            self.noise_w * noise_loss
        )

        return total, {
            'mae':   mae.item(),
            'ssim':  ssim_loss.item(),
            'grad':  grad_loss.item(),
            'noise': noise_loss.item(),
        }

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dy  = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
        pred_dx  = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
        tgt_dy   = target[:, :, 1:, :] - target[:, :, :-1, :]
        tgt_dx   = target[:, :, :, 1:] - target[:, :, :, :-1]
        return F.l1_loss(pred_dy, tgt_dy) + F.l1_loss(pred_dx, tgt_dx)


# ── Training Loop ────────────────────────────────────────────────────────────

def train(args):
    # ── Device ──
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nDevice: {device}")
    if device == 'cpu':
        print("  [WARNING] CPU training will be slow. Consider using a GPU.")

    # ── Dataset ──
    DatasetClass = AstroDatasetPreloaded if args.preload else AstroDataset
    full_dataset = DatasetClass(
        data_root=args.data_root,
        patch_size=args.patch_size,
        patches_per_sub=args.patches_per_sub,
        in_channels=args.channels,
        augment=True,
    )

    # 90/10 train/val split
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train patches: {train_size}  |  Val patches: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device == 'cuda')
    )

    # ── Model ──
    model = build_model(in_channels=args.channels, device=device, num_subs=args.num_subs)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── Optimiser & Scheduler ──
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = AstroNoiseLoss(
        mae_weight=1.0,
        ssim_weight=1.0,
        grad_weight=0.4,
        noise_weight=0.3,
        channels=args.channels,
    ).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Epoch Loop ──
    print(f"\nStarting training for {args.epochs} epochs\n{'='*60}")
    for epoch in range(start_epoch, args.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for sub, stack in train_loader:
            sub = sub.to(device)
            stack = stack.to(device)
            optimiser.zero_grad()
            pred = model(sub)
            loss, _ = criterion(pred, stack)
            loss.backward()
            # Gradient clipping prevents instability on bright star gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_components = {'mae': 0.0, 'ssim': 0.0, 'grad': 0.0, 'noise': 0.0}
        with torch.no_grad():
            for sub, stack in val_loader:
                sub, stack = sub.to(device), stack.to(device)
                pred = model(sub)
                loss, comps = criterion(pred, stack)
                val_loss += loss.item()
                for k in val_components:
                    val_components[k] += comps[k]

        val_loss /= len(val_loader)
        for k in val_components:
            val_components[k] /= len(val_loader)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:4d}/{args.epochs} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} "
            f"(MAE={val_components['mae']:.4f} SSIM={val_components['ssim']:.4f} "
            f"Grad={val_components['grad']:.4f} Noise={val_components['noise']:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoint ──
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimiser_state': optimiser.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }
            ckpt_path = out_dir / f'checkpoint_epoch{epoch+1:04d}.pt'
            torch.save(ckpt, ckpt_path)
            if is_best:
                best_path = out_dir / 'best_model.pt'
                torch.save(ckpt, best_path)
                print(f"  ★ New best model saved → {best_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    print(f"Best model saved to: {out_dir / 'best_model.pt'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train AstroDeNoise noise reduction model')
    p.add_argument('--data_root',       default='./data',         help='Root directory of training data')
    p.add_argument('--out_dir',         default='./models',       help='Directory for model checkpoints')
    p.add_argument('--resume',          default='',               help='Path to checkpoint to resume from')
    p.add_argument('--epochs',          type=int, default=100,    help='Number of training epochs')
    p.add_argument('--batch_size',      type=int, default=4,      help='Batch size')
    p.add_argument('--patch_size',      type=int, default=256,    help='Training patch size (pixels)')
    p.add_argument('--patches_per_sub', type=int, default=8,      help='Random patches extracted per sub per epoch')
    p.add_argument('--channels',        type=int, default=3,      choices=[1, 3], help='1=mono, 3=OSC colour')
    p.add_argument('--lr',              type=float, default=1e-4, help='Initial learning rate')
    p.add_argument('--workers',         type=int, default=4,      help='DataLoader worker threads')
    p.add_argument('--device',          default='auto',           help='cuda / cpu / auto')
    p.add_argument('--preload',         action='store_true',      help='Preload all images into RAM')
    p.add_argument('--num_subs',        type=int, default=0,      help='Total subs across all targets (auto-scales model size and dropout)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
