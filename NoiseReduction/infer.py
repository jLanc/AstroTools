"""
AstroDeNoise Inference Script

Processes a full linear XISF or FITS image using the trained noise reduction model.
Uses tiled inference with Gaussian-weighted overlap to avoid tile-boundary artefacts
— essential for large astrophotography images (e.g. 6248×4176 from ASI2600MC Pro).

Usage:
    python infer.py --model ./models/best_model.pt --input ./my_image.xisf --output ./denoised.xisf

Input must be a calibrated, linear (un-stretched) image.
It can be a single noisy sub or an already-integrated master frame.
The output is a linear XISF file — do NOT stretch before passing in.
Stretch the denoised output in PixInsight as you normally would.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model import build_model
from dataset import load_image_linear, save_image_linear


def infer_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
    device: str = 'cpu',
    batch_size: int = 1,
) -> np.ndarray:
    """
    Run tiled inference on a full image with Gaussian-weighted blending.

    Why Gaussian instead of Hanning?
    The Hanning window reaches exactly zero at its edges. In the overlap zone,
    near-edge pixels have near-zero weight from both adjacent tiles — the
    normalisation rescues the value but the tiny absolute weights amplify any
    model discontinuity into a visible seam. A Gaussian window never reaches
    zero, so every pixel in the overlap zone always has meaningful weight from
    each tile that covers it, producing a smooth, natural blend.

    overlap should be at least tile_size // 4 (default 128 for a 512px tile).
    Larger overlap = smoother seams, slower processing.

    Args:
        image:      (C, H, W) float32, normalised to [0, 1]
        tile_size:  Processing tile size (match training patch_size)
        overlap:    Overlap in pixels between adjacent tiles
        batch_size: Process this many tiles at once
    """
    model.eval()
    C, H, W = image.shape
    output     = np.zeros((C, H, W), dtype=np.float32)
    weight_map = np.zeros((H, W),    dtype=np.float32)

    stride = tile_size - overlap

    # Gaussian blend window — sigma chosen so the window is ~0.01 at the very edge,
    # giving strong centre-weighting while never reaching zero.
    sigma   = tile_size / 5.0
    coords  = np.arange(tile_size) - (tile_size - 1) / 2.0
    gauss1d = np.exp(-0.5 * (coords / sigma) ** 2).astype(np.float32)
    blend2d = np.outer(gauss1d, gauss1d)   # (tile_size, tile_size)

    # Tile coordinate grid — always cover full image edges
    ys = list(range(0, max(1, H - tile_size), stride))
    xs = list(range(0, max(1, W - tile_size), stride))
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(0, W - tile_size))

    coords_list = [(y, x) for y in ys for x in xs]
    total_tiles = len(coords_list)
    print(f"  Tiled inference: {total_tiles} tiles ({len(ys)}×{len(xs)}), "
          f"tile={tile_size}px, overlap={overlap}px, blend=gaussian")

    for batch_start in range(0, total_tiles, batch_size):
        batch_coords = coords_list[batch_start:batch_start + batch_size]
        tiles = [image[:, y:y + tile_size, x:x + tile_size] for y, x in batch_coords]

        batch_tensor = torch.from_numpy(np.stack(tiles)).to(device)
        with torch.no_grad():
            enhanced_batch = model(batch_tensor).cpu().numpy()

        for i, (y, x) in enumerate(batch_coords):
            for c in range(C):
                output[c, y:y + tile_size, x:x + tile_size] += enhanced_batch[i, c] * blend2d
            weight_map[y:y + tile_size, x:x + tile_size] += blend2d

        if (batch_start // batch_size) % 20 == 0:
            print(f"    {100 * batch_start / total_tiles:.0f}% ({batch_start}/{total_tiles})")

    output /= np.maximum(weight_map[np.newaxis], 1e-8)
    output  = np.clip(output, 0.0, 1.0)
    print(f"  Done. Output range: [{output.min():.4f}, {output.max():.4f}]")
    return output


def load_model(model_path: str, device: str) -> tuple:
    """Load a trained model checkpoint and return (model, args_dict)."""
    ckpt = torch.load(model_path, map_location=device)
    args = ckpt.get('args', {})
    in_channels = args.get('channels', 3)
    model = build_model(in_channels=in_channels, device=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, args


def process_image(
    model_path: str,
    input_path: str,
    output_path: str,
    tile_size: int = 512,
    overlap: int = 64,
    device: str = 'auto',
    batch_size: int = 1,
):
    """Full pipeline: load → infer → save."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {model_path}")
    model, train_args = load_model(model_path, device)
    in_channels = train_args.get('channels', 3)
    print(f"  Trained with {in_channels}-channel ({'colour' if in_channels==3 else 'mono'}) data")

    # Load input image
    input_path = Path(input_path)
    output_path = Path(output_path)
    print(f"\nLoading: {input_path.name}  ({input_path.suffix.upper()})")
    image, norm_constant = load_image_linear(input_path, in_channels)
    print(f"  Shape: {image.shape}  |  Norm constant: {norm_constant:.2f}")

    print(f"\nRunning tiled inference (noise reduction)...")
    enhanced = infer_tiled(
        model, image,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
        batch_size=batch_size,
    )

    # Save — default output format is XISF; use .fits extension to save as FITS instead
    # Source path is passed through so XISF metadata (camera, filter, etc.) is preserved
    save_image_linear(enhanced, output_path, norm_constant, source_path=input_path)
    print(f"\nSaved denoised image → {output_path}")
    print("  (Still linear data — stretch in PixInsight as normal)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Run AstroDeNoise inference on a linear XISF/FITS image')
    p.add_argument('--model',       required=True,          help='Path to trained model .pt file')
    p.add_argument('--input',       required=True,          help='Input FITS file (linear, calibrated)')
    p.add_argument('--output',      required=True,          help='Output FITS file path')
    p.add_argument('--tile_size',   type=int, default=512,  help='Inference tile size (match training patch_size)')
    p.add_argument('--overlap',     type=int, default=128,  help='Tile overlap in pixels (>=tile_size//4 recommended)')
    p.add_argument('--batch_size',  type=int, default=1,    help='Tiles per GPU batch (increase if VRAM allows)')
    p.add_argument('--device',      default='auto',         help='cuda / cpu / auto')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_image(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
        batch_size=args.batch_size,
    )
