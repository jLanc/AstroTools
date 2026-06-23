# AstroEnhance — AI Noise Reduction for Linear Astrophotography

A PyTorch-based U-Net trained on calibrated sub → master stack pairs. The model
learns your camera's noise profile and can denoise either new subs before
integration or an already-stacked master frame.

---

## How the training methodology works

The model is trained on pairs of images from the same dataset:

- **Input:** a calibrated, linear noisy sub
- **Target:** the master integration of all subs — the final clean stack

This trains the network to predict a clean linear image suitable for stretch
later, while preserving real signal and removing read noise, shot noise and
thermal patterns.

Once trained, the model can be applied to new noisy subs before stacking or
run directly on an already-integrated master frame for additional cleanup.

**Why this is better than a generic noise reduction tool:**
Most denoising algorithms (median filter, wavelets, etc.) don't know what noise
looks like for your specific camera at your specific gain and temperature settings.
This model trains exclusively on your data, so it learns the exact read noise
profile, thermal pattern, and sky background characteristics of your equipment.

**Key requirements:**
1. Subs must be **calibrated** (bias/dark/flat applied) before training
2. Subs must be **registered** to the stack (PixInsight StarAlignment)
3. Everything must stay **linear** — do not stretch before inference
4. Subs and stack must be exported as **XISF** from PixInsight (or FITS)
5. More targets and more diverse data = better generalisation

---

## Project structure

```
NoiseReduction/
├── model.py          # U-Net architecture with residual blocks and dropout
├── dataset.py        # XISF/FITS loading, normalisation, patch extraction
├── train.py          # Training loop with noise-reduction loss function
├── infer.py          # Tiled inference for full-resolution images
├── AstroDenoise.js   # PixInsight PJSR plugin bridge
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Installation

### 1. Python environment

Use a virtual environment for the project.

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Git Bash:
```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the core dependencies:
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For Windows with NVIDIA CUDA 12.1, run:
```bash
python -m pip install -r requirements-windows-cuda.txt
```

If you need PixInsight XISF support, install the optional package:
```bash
python -m pip install -r requirements-optional.txt
```

If `xisf` is unavailable for your Python/OS, use FITS files instead; `astropy` is already included.

### Supported Python versions

- macOS / Linux: Python 3.10–3.14
- Windows CUDA: Python 3.11 or 3.12 (recommended)

### Notes for Windows

- Do not use Python 3.14 with CUDA 12.1 wheels on Windows.
- In Git Bash, use `source .venv/Scripts/activate` instead of PowerShell activation commands.
- The GPU CUDA install path only affects `torch` / `torchvision` / `torchaudio`.

### Conda alternative

```bash
conda create -n astrodenoise python=3.11 -y
conda activate astrodenoise
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### PixInsight plugin

Edit the path variables at the top of `AstroDenoise.js`:
```javascript
var PYTHON_PATH   = "/path/to/.venv/Scripts/python.exe";
var INFER_SCRIPT  = "/path/to/NoiseReduction/infer.py";
var DEFAULT_MODEL = "/path/to/models/best_model.pt";
```

Then add the script in PixInsight: **Script → Feature Scripts → Add**.

---

## Preparing your data

Edit the three path variables at the top of `AstroEnhance.js`:
```javascript
var PYTHON_PATH   = "/path/to/astro_env/bin/python";
var INFER_SCRIPT  = "/path/to/astro_enhance/infer.py";
var DEFAULT_MODEL = "/path/to/models/best_model.pt";
```

Then in PixInsight: **Script → Feature Scripts → Add** → select `AstroEnhance.js`

---

## Preparing your data

### Directory structure

```
data/
├── eta_carinae/
│   ├── subs/
│   │   ├── Light_001.xisf
│   │   ├── Light_002.xisf
│   │   └── ...              ← calibrated, registered subs
│   └── master/
│       └── master.xisf      ← your full integration of all subs
├── orion/
│   ├── subs/
│   └── master/
└── ...                      ← one directory per target
```

### PixInsight workflow before training

1. **Calibrate** all subs with ImageCalibration (bias, dark, flat)
2. **Integrate** all subs into a master stack with ImageIntegration
3. **Register** all subs to the master stack with StarAlignment
   — use the master stack as the reference frame
4. **Export** subs and stack as 32-bit float XISF (File → Save As)
   — do not apply any stretch, curves, or processing

### Camera settings

| Camera | Resolution | Channels flag |
|---|---|---|
| ASI2600MC Pro | 6248×4176 | `--channels 3` |
| ASI294MC Pro | 4144×2822 | `--channels 3` |

Both cameras output colour (OSC) data — always use `--channels 3` with debayered RGB.

### Dataset scale and model sizing

The model auto-scales to your dataset. Pass your approximate total sub count
with `--num_subs` and it selects the right capacity and dropout automatically:

| Total subs | Model config | Parameters |
|---|---|---|
| < 50 | base_features=24, dropout=0.25 | ~9M |
| 50–200 | base_features=32, dropout=0.10 | ~27M |
| 200–500 | base_features=48, dropout=0.05 | ~60M |
| 500+ | base_features=64, dropout=0.00 | ~108M |

More parameters with too little data causes the model to memorise noise
patterns rather than learn to remove them. Let `--num_subs` handle this.

---

## Training

```bash
# Small test run — one target, few subs
python train.py \
    --data_root ./data \
    --out_dir   ./models \
    --num_subs  10 \
    --channels  3 \
    --epochs    150 \
    --device    mps        # or cuda / cpu

# Full training run — many targets
python train.py \
    --data_root      ./data \
    --out_dir        ./models \
    --num_subs       6000 \
    --channels       3 \
    --epochs         150 \
    --batch_size     8 \
    --patch_size     256 \
    --patches_per_sub 16 \
    --device         mps

# Resume interrupted training
python train.py \
    --data_root ./data \
    --resume    ./models/checkpoint_epoch0080.pt \
    --num_subs  6000 \
    --epochs    150
```

**Do not use `--preload`** unless you have 64GB+ of RAM available. With 60+
datasets of large XISF files the preloaded cache will exhaust memory before
the GPU gets a chance to use it.

### Expected training time

| Hardware | 150 epochs, ~500 subs |
|---|---|
| M1 (8-core GPU, MPS) | ~8–12 hours |
| M1 Pro / M1 Max | ~4–6 hours |
| RTX 3080 | ~2–3 hours |
| RTX 4090 | ~1–1.5 hours |
| CPU only | 24–48 hours |

### Reading the training output

```
Epoch   1/150 | Train: 0.28431 | Val: 0.30102 | (MAE=0.0142 SSIM=0.2184 Grad=0.0412 Noise=0.0014)
Epoch  10/150 | Train: 0.09823 | Val: 0.10541 | ...
Epoch  60/150 | Train: 0.03201 | Val: 0.03887 | ★ best model saved
Epoch 150/150 | Train: 0.01823 | Val: 0.04102 |   ← gap widening, overfit zone
```

Use `best_model.pt`, not the final epoch checkpoint.

**What each loss component means:**

| Component | What it measures | Good sign |
|---|---|---|
| MAE | Raw pixel accuracy | Steadily falling |
| SSIM | Structural similarity / noise penalty | Falling toward 0 |
| Grad | Edge sharpness preservation | Low and stable |
| Noise | Residual noise in smooth regions | Falling toward 0 |

**SSIM** is the most important number to watch for noise reduction quality.
If it plateaus early, try a lower learning rate (`--lr 0.00005`).

---

## Running inference

### From the command line

```bash
python infer.py \
    --model     ./models/best_model.pt \
    --input     ./linear_image.xisf \
    --output    ./denoised_image.xisf \
    --tile_size 512 \
    --overlap   128 \
    --device    mps
```

The input can be a calibrated noisy sub or an already-integrated master frame.
The output is a linear XISF file. Stretch the denoised result in PixInsight
only after processing.

### From PixInsight

1. Open a linear XISF image (do NOT apply STF or any stretch first)
2. Run **Script → AstroEnhance → AstroEnhance**
3. Select your `best_model.pt` file
4. Click "Enhance"
5. The denoised image opens as `ImageName_AstroEnhanced`

The plugin checks the image median before processing and will warn you if the
image appears to be stretched (median > 0.1). Linear calibrated data typically
has a median well below 0.05.

### Recommended workflow

```
Calibrated linear subs
        │
        ▼
[AstroEnhance — denoise each sub]
        │
        ▼
Denoised linear subs  ──►  ImageIntegration (stack as normal)
        │
        ▼
Denoised stack  ──►  Stretch in PixInsight
```

For the best integration results, train the model with noisy subs and the
master stack as the target. During inference you may denoise new subs before
stacking or run the same trained model on a stacked master frame to clean up
residual noise in the final integration.

---

## Tips for best results

**Data quality matters more than quantity:**
- Deep subs (5+ minutes) have more real signal relative to noise — the model
  has a clearer learning signal
- Mix targets across different object types: emission nebulae, galaxies, IFN
  fields, star clusters — diversity prevents the model overfitting to one sky
- Mix both cameras if possible — the model learns each camera's noise profile

**What this tool does:**
- Removes read noise, shot noise, and thermal signal from linear subs
- Preserves star morphology and genuine faint structure
- Learns your specific camera's noise characteristics from your own data

**What this tool does NOT do:**
- It is not a deconvolution tool and will not sharpen stars
- It will not recover detail that wasn't recorded — very short subs with
  essentially no signal above the noise floor won't benefit much
- It is not a replacement for calibration frames — always calibrate first

---

## Troubleshooting

**"Shape mismatch" error during training:**
Your subs are not aligned to the stack. Run PixInsight StarAlignment with
the master stack as the reference frame before exporting.

**"Image appears to be all zeros" error:**
Negative pixel values from sky background subtraction or aggressive calibration
are being clipped. Check your calibration frames and ImageCalibration settings.

**Tile seams visible in output:**
Increase `--overlap` (default 128). Try 192 or 256 on a 512px tile. Also
ensure you're running the latest `infer.py` which uses Gaussian blending.

**PixInsight plugin fails (exit code ≠ 0):**
Run `infer.py` manually from the terminal first to see the actual error.
The most common cause is `PYTHON_PATH` pointing to a Python installation
that doesn't have `torch` or `xisf` installed.

**Training loss not moving after epoch 5:**
Learning rate may be too high. Try `--lr 0.00005`. Also verify your data is
genuinely linear — if the median pixel value across your subs is above 0.1
the data has likely been stretched before export.

**SSIM loss stuck above 0.1:**
This usually means the model is struggling to distinguish noise from real
signal, often because subs are too short or too few. Try adding more data
or using deeper subs from the same targets.
