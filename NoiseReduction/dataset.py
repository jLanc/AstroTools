"""
AstroDeNoise Dataset — loads (single_sub, master_stack) training pairs.

Supports XISF (PixInsight native) and FITS formats.
XISF is the default — that's what PixInsight's StarAlignment and
ImageIntegration output natively.

Directory layout expected:
    data/
        target_orion/
            subs/
                sub_001.xisf
                sub_002.xisf
                ...
            stack/
                master_stack.xisf    ← your integrated/stacked image
        target_carina/
            subs/
                ...
            stack/
                master_stack.xisf

Both the subs and the stack MUST be:
  - Calibrated (bias/dark/flat applied)
  - In linear (un-stretched) space
  - Debayered to RGB (or kept mono — set in_channels accordingly)
  - Registered/aligned to each other (use PixInsight StarAlignment first)

Data augmentation applied during training:
  - Random flips (horizontal / vertical) — safe for astronomy, stars don't care
  - Random 90° rotations — same reason
  - Random patch crops — allows large images to yield many training samples
  - NO intensity augmentation — we're in linear space, real signal ratios must be preserved
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Format support ──────────────────────────────────────────────────────────
# XISF is PixInsight's native format — preferred for all PI-exported data.
# FITS is kept as a fallback.

try:
    from xisf import XISF
    _XISF_AVAILABLE = True
except ImportError:
    _XISF_AVAILABLE = False
    print("[WARNING] xisf package not found. Install with: pip install xisf")

try:
    from astropy.io import fits as astropy_fits
    _FITS_AVAILABLE = True
except ImportError:
    _FITS_AVAILABLE = False

if not _XISF_AVAILABLE and not _FITS_AVAILABLE:
    raise ImportError("Install at least one image format library: pip install xisf astropy")

# File extensions recognised per format
_XISF_EXTS = {'.xisf'}
_FITS_EXTS  = {'.fits', '.fit', '.fts'}
_ALL_EXTS   = _XISF_EXTS | _FITS_EXTS


# ── Unified image loader ────────────────────────────────────────────────────

def load_image_linear(
    path: Path,
    expected_channels: int = 3,
    norm_constant: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Load a XISF or FITS image into a (C, H, W) float32 array normalised to [0, 1].

    Args:
        path:             Image file path (.xisf, .fits, .fit)
        expected_channels: 1 for mono, 3 for OSC colour
        norm_constant:    If provided, normalise by THIS value instead of computing
                          the image's own 99.9th percentile. Pass the stack's norm
                          constant when loading a sub so the pair shares the same
                          linear scale — this is critical for correct training.

    XISF loading:
      - Uses xisf.XISF.read() → (H, W, C), transposed to (C, H, W)
    FITS loading:
      - Handles PixInsight (C, H, W) and HWC layouts, and 2D mono

    Returns:
        (array, norm_constant_used)
        The norm constant is needed by save_image_linear to restore ADU range.
    """
    ext = path.suffix.lower()

    if ext in _XISF_EXTS:
        if not _XISF_AVAILABLE:
            raise RuntimeError(f"Cannot load {path.name}: xisf package not installed")
        data = _load_xisf(path)
    elif ext in _FITS_EXTS:
        if not _FITS_AVAILABLE:
            raise RuntimeError(f"Cannot load {path.name}: astropy package not installed")
        data = _load_fits(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}  (supported: {_ALL_EXTS})")

    data = data.astype(np.float32)

    # ── Ensure (C, H, W) ──
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        if data.shape[2] in (1, 3) and data.shape[0] not in (1, 3):
            data = np.transpose(data, (2, 0, 1))

    # ── Channel count check ──
    c = data.shape[0]
    if c != expected_channels:
        if c == 1 and expected_channels == 3:
            data = np.repeat(data, 3, axis=0)
            print(f"  [WARNING] {path.name}: mono expanded to 3-channel duplicate")
        else:
            raise ValueError(
                f"{path.name}: expected {expected_channels} channels, got {c}"
            )

    data = np.clip(data, 0.0, None)

    # Use provided norm constant, or compute from this image's own statistics
    if norm_constant is None:
        norm_constant = float(np.percentile(data, 99.9))
        if norm_constant <= 0:
            raise ValueError(f"{path.name}: image appears to be all zeros — check calibration")

    data /= norm_constant
    # Clip to [0,1] in case any pixel slightly exceeded the norm constant
    data = np.clip(data, 0.0, 1.0)

    return data, norm_constant


def _load_xisf(path: Path) -> np.ndarray:
    """Load XISF file, return raw numpy array in (C, H, W) layout."""
    image_meta = {}
    data = XISF.read(str(path), image_metadata=image_meta)
    # XISF.read returns channels_last (H, W, C) by default
    # Transpose to (C, H, W) to match our convention
    if data.ndim == 3 and data.shape[2] in (1, 3):
        data = np.transpose(data, (2, 0, 1))
    return data


def _load_fits(path: Path) -> np.ndarray:
    """Load FITS file, return raw numpy array."""
    with astropy_fits.open(str(path)) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.data.copy()
    raise ValueError(f"No image data found in {path}")


# ── Unified image saver ─────────────────────────────────────────────────────

def save_image_linear(
    array: np.ndarray,
    out_path: Path,
    norm_constant: float = 1.0,
    source_path: Optional[Path] = None,
):
    """
    Save a (C, H, W) float32 array as XISF (or FITS if .fit/.fits extension given).
    Multiplies by norm_constant to restore original ADU range before saving.

    If source_path is provided and was a XISF file, its embedded metadata
    (FITSKeywords, XISFProperties like telescope, camera, filter, etc.)
    is copied across to the output — PixInsight will see the full capture metadata.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = (array * norm_constant).astype(np.float32)

    ext = out_path.suffix.lower()

    if ext in _XISF_EXTS or ext not in _FITS_EXTS:
        # Default: save as XISF
        if not _XISF_AVAILABLE:
            raise RuntimeError("Cannot save XISF: xisf package not installed")

        # Carry forward source metadata if available
        image_metadata = {}
        if source_path is not None and source_path.suffix.lower() in _XISF_EXTS:
            try:
                src_meta = {}
                XISF.read(str(source_path), image_metadata=src_meta)
                # Only copy the safe metadata keys — not geometry (that changes)
                for key in ('FITSKeywords', 'XISFProperties'):
                    if key in src_meta:
                        image_metadata[key] = src_meta[key]
            except Exception:
                pass  # metadata copy is best-effort

        # XISF.write expects (H, W, C) channels-last layout
        out_hwc = np.transpose(out, (1, 2, 0)) if out.ndim == 3 else out
        XISF.write(
            str(out_path),
            out_hwc,
            creator_app="AstroDeNoise AI",
            image_metadata=image_metadata,
        )
    else:
        # FITS fallback
        if not _FITS_AVAILABLE:
            raise RuntimeError("Cannot save FITS: astropy not installed")
        hdr = astropy_fits.Header()
        hdr['COMMENT'] = 'Processed by AstroDeNoise AI'
        hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(out, header=hdr)])
        hdul.writeto(str(out_path), overwrite=True)


# Backwards-compatible alias for any code that still references the old name
def load_fits_linear(path, expected_channels=3):
    return load_image_linear(path, expected_channels)

def save_fits_linear(array, out_path, norm_constant=1.0, header_template=None):
    save_image_linear(array, out_path, norm_constant)


# ── Dataset ─────────────────────────────────────────────────────────────────

class AstroDataset(Dataset):
    """
    Training dataset of (sub_patch, stack_patch) pairs.
    
    Args:
        data_root:       Root directory containing per-target subdirectories
        patch_size:      Training patch size in pixels (must be divisible by 2^depth)
        patches_per_sub: How many random patches to extract per sub per epoch
        in_channels:     1 = mono, 3 = OSC colour
        augment:         Whether to apply flip/rotate augmentation
    """
    def __init__(
        self,
        data_root: str,
        patch_size: int = 256,
        patches_per_sub: int = 8,
        in_channels: int = 3,
        augment: bool = True,
    ):
        self.patch_size = patch_size
        self.patches_per_sub = patches_per_sub
        self.in_channels = in_channels
        self.augment = augment
        self.pairs: List[Tuple[Path, Path]] = []

        data_root = Path(data_root)
        self._discover_pairs(data_root)

        if not self.pairs:
            raise FileNotFoundError(
                f"No (sub, stack) pairs found under {data_root}. "
                "Expected structure: target_name/subs/*.xisf|.fits and target_name/master/master_stack.xisf|.fits "
                "(or the legacy target_name/stack/ folder)."
            )
        print(f"Dataset: {len(self.pairs)} sub-stack pairs from {data_root}")

    def _discover_pairs(self, root: Path):
        """Walk directory tree to find all sub ↔ stack pairs."""
        for target_dir in sorted(root.iterdir()):
            if not target_dir.is_dir():
                continue
            subs_dir = target_dir / 'subs'
            master_dir = target_dir / 'master'
            stack_dir = target_dir / 'stack'
            if not subs_dir.exists() or (not master_dir.exists() and not stack_dir.exists()):
                continue

            image_dir = master_dir if master_dir.exists() else stack_dir

            # Find master stack — prefer XISF, fall back to FITS
            stack_files = (
                sorted(image_dir.glob('*.xisf')) +
                sorted(image_dir.glob('*.fits')) +
                sorted(image_dir.glob('*.fit'))
            )
            if not stack_files:
                print(f"  [WARN] No stack/master image found in {image_dir}, skipping.")
                continue
            master_stack = stack_files[0]

            # Pair every sub with this stack — prefer XISF, fall back to FITS
            sub_files = (
                sorted(subs_dir.glob('*.xisf')) +
                sorted(subs_dir.glob('*.fits')) +
                sorted(subs_dir.glob('*.fit'))
            )
            if not sub_files:
                print(f"  [WARN] No sub FITS found in {subs_dir}, skipping.")
                continue

            for sub in sub_files:
                self.pairs.append((sub, master_stack))

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_sub

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx = idx // self.patches_per_sub
        sub_path, stack_path = self.pairs[pair_idx]

        # ── Critical: load stack first to get its norm constant,
        # then normalise the sub using that SAME constant.
        #
        # Why this matters: the stack has higher SNR so its 99.9th percentile
        # is higher than a single sub. If each is normalised independently,
        # both end up in [0,1] but the ratio between them is destroyed —
        # the model has no way to learn the correct sub→stack signal mapping.
        # Sharing the stack's norm constant preserves the linear relationship.
        stack_arr, stack_norm = load_image_linear(stack_path, self.in_channels)
        sub_arr, _            = load_image_linear(sub_path,   self.in_channels,
                                                  norm_constant=stack_norm)

        if sub_arr.shape[1:] != stack_arr.shape[1:]:
            raise ValueError(
                f"Shape mismatch: sub {sub_arr.shape} vs stack {stack_arr.shape} "
                f"— please register your subs to the stack in PixInsight first."
            )

        H, W = sub_arr.shape[1], sub_arr.shape[2]
        ps = self.patch_size

        if H < ps or W < ps:
            raise ValueError(
                f"Image {sub_path.name} ({H}x{W}) is smaller than patch_size={ps}. "
                "Reduce patch_size or use larger images."
            )

        # Random crop — same patch location in both images (they're registered)
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        sub_patch = sub_arr[:, top:top + ps, left:left + ps]
        stack_patch = stack_arr[:, top:top + ps, left:left + ps]

        # Augmentation: random flip/rotate (same transform applied to both)
        if self.augment:
            sub_patch, stack_patch = self._augment(sub_patch, stack_patch)

        return (
            torch.from_numpy(sub_patch.copy()),
            torch.from_numpy(stack_patch.copy()),
        )

    def _augment(self, sub: np.ndarray, stack: np.ndarray):
        """Apply identical random geometric transforms to sub and stack."""
        # Horizontal flip
        if random.random() > 0.5:
            sub = sub[:, :, ::-1]
            stack = stack[:, :, ::-1]
        # Vertical flip
        if random.random() > 0.5:
            sub = sub[:, ::-1, :]
            stack = stack[:, ::-1, :]
        # 90° rotation (k in {0,1,2,3})
        k = random.randint(0, 3)
        if k > 0:
            sub = np.rot90(sub, k, axes=(1, 2))
            stack = np.rot90(stack, k, axes=(1, 2))
        return sub, stack


class AstroDatasetPreloaded(AstroDataset):
    """
    Variant that preloads all images into RAM.
    Faster training if you have enough memory (16 GB+ recommended for typical datasets).

    Cache stores (array, norm_constant) per path. Sub arrays are re-normalised
    using their paired stack's norm constant so the linear signal ratio is preserved.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Preloading dataset into RAM...")

        # Load all stacks first (we need their norm constants)
        self._stack_cache: dict = {}
        stack_paths = set(stack for _, stack in self.pairs)
        for path in sorted(stack_paths):
            self._stack_cache[path] = load_image_linear(path, self.in_channels)
            print(f"  Stack: {path.name}  norm={self._stack_cache[path][1]:.4f}")

        # Load subs, normalising each by its paired stack's norm constant
        # (stack_norm is the same for all subs in the same target directory)
        self._sub_cache: dict = {}
        for sub_path, stack_path in self.pairs:
            if sub_path not in self._sub_cache:
                _, stack_norm = self._stack_cache[stack_path]
                self._sub_cache[sub_path] = load_image_linear(
                    sub_path, self.in_channels, norm_constant=stack_norm
                )

        print(f"  Loaded {len(self._stack_cache)} stacks, {len(self._sub_cache)} subs.")

    def __getitem__(self, idx):
        pair_idx = idx // self.patches_per_sub
        sub_path, stack_path = self.pairs[pair_idx]
        sub_arr,   _ = self._sub_cache[sub_path]
        stack_arr, _ = self._stack_cache[stack_path]

        H, W = sub_arr.shape[1], sub_arr.shape[2]
        ps = self.patch_size
        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        sub_patch   = sub_arr[:,   top:top + ps, left:left + ps]
        stack_patch = stack_arr[:, top:top + ps, left:left + ps]
        if self.augment:
            sub_patch, stack_patch = self._augment(sub_patch, stack_patch)
        return (
            torch.from_numpy(sub_patch.copy()),
            torch.from_numpy(stack_patch.copy()),
        )
