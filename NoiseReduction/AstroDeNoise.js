// AstroDeNoise PixInsight Plugin
// PixInsight JavaScript Runtime (PJSR) bridge to the Python inference backend.
//
// INSTALLATION:
//   1. Copy this file to:
//      [PixInsight]/src/scripts/AstroDeNoise/AstroDeNoise.js
//   2. In PixInsight: Script → Feature Scripts → Add → select this file
//   3. Configure PYTHON_PATH and INFER_SCRIPT below to match your system
//
// USAGE:
//   - Open your linear FITS image in PixInsight (do NOT stretch)
//   - Run Script → AstroDeNoise → AstroDeNoise
//   - Select your model .pt file
//   - Click "Denoise"
//   - The enhanced image opens as a new image view

#feature-id    AstroDeNoise : Scripts > AstroDeNoise
#feature-info  AI-powered noise reduction for linear astrophotography data.

// ── User Configuration ─────────────────────────────────────────────────────
// Adjust these paths for your system:

// Path to python executable (use full path to your venv if applicable)
// Windows example: "C:/Users/Jake/AppData/Local/Programs/Python/Python311/python.exe"
// macOS/Linux example: "/usr/bin/python3" or "/home/jake/venv/bin/python"
var PYTHON_PATH = "python3";

// Path to your AstroDeNoise infer.py script
var INFER_SCRIPT = "/path/to/astro_denoise/infer.py";

// Default model path (can be changed in the dialog)
var DEFAULT_MODEL = "/path/to/models/best_model.pt"; // trained noise reduction model

// Tile size for inference (match your training patch_size)
var TILE_SIZE = 512;

// Tile overlap (larger = smoother, slower)
var OVERLAP = 64;

// ── Dialog ─────────────────────────────────────────────────────────────────

function AstroDeNoiseDialog() {
    this.__base__ = Dialog;
    this.__base__();

    this.windowTitle = "AstroDeNoise — AI Noise Reduction";

    // Model path label + edit + browse button
    this.modelLabel = new Label(this);
    this.modelLabel.text = "Model (.pt):";
    this.modelLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

    this.modelEdit = new Edit(this);
    this.modelEdit.text = DEFAULT_MODEL;
    this.modelEdit.setFixedWidth(400);

    this.modelBrowse = new ToolButton(this);
    this.modelBrowse.icon = this.scaledResource(":/icons/open.png");
    this.modelBrowse.toolTip = "Browse for model file";
    this.modelBrowse.onClick = function() {
        var fd = new OpenFileDialog();
        fd.caption = "Select AstroDeNoise Model";
        fd.filters = [["PyTorch Model", "*.pt"], ["All Files", "*"]];
        if (fd.execute()) {
            this.dialog.modelEdit.text = fd.fileName;
        }
    };

    this.modelSizer = new HorizontalSizer;
    this.modelSizer.spacing = 4;
    this.modelSizer.add(this.modelLabel);
    this.modelSizer.add(this.modelEdit, 1);
    this.modelSizer.add(this.modelBrowse);

    // Tile size slider
    this.tileSizeLabel = new Label(this);
    this.tileSizeLabel.text = "Tile size:";

    this.tileSizeSpinBox = new SpinBox(this);
    this.tileSizeSpinBox.setRange(128, 1024);
    this.tileSizeSpinBox.stepSize = 64;
    this.tileSizeSpinBox.value = TILE_SIZE;

    this.tileSizeSizer = new HorizontalSizer;
    this.tileSizeSizer.spacing = 4;
    this.tileSizeSizer.add(this.tileSizeLabel);
    this.tileSizeSizer.add(this.tileSizeSpinBox);
    this.tileSizeSizer.addStretch();

    // Info label
    this.infoLabel = new Label(this);
    this.infoLabel.text =
        "Process the active image view with the AstroDeNoise AI model.\n" +
        "Input must be a calibrated, linear (un-stretched) FITS image.\n" +
        "Output will open as a new image view.";
    this.infoLabel.wordWrapping = true;

    // OK / Cancel buttons
    this.okButton = new PushButton(this);
    this.okButton.text = "Denoise";
    this.okButton.icon = this.scaledResource(":/icons/ok.png");
    this.okButton.onClick = function() {
        this.dialog.ok();
    };

    this.cancelButton = new PushButton(this);
    this.cancelButton.text = "Cancel";
    this.cancelButton.icon = this.scaledResource(":/icons/cancel.png");
    this.cancelButton.onClick = function() {
        this.dialog.cancel();
    };

    this.buttonSizer = new HorizontalSizer;
    this.buttonSizer.addStretch();
    this.buttonSizer.add(this.okButton);
    this.buttonSizer.addSpacing(8);
    this.buttonSizer.add(this.cancelButton);

    // Main layout
    this.sizer = new VerticalSizer;
    this.sizer.margin = 8;
    this.sizer.spacing = 6;
    this.sizer.add(this.infoLabel);
    this.sizer.addSpacing(8);
    this.sizer.add(this.modelSizer);
    this.sizer.add(this.tileSizeSizer);
    this.sizer.addStretch();
    this.sizer.add(this.buttonSizer);

    this.adjustToContents();
}
AstroDeNoiseDialog.prototype = new Dialog;


// ── Main Execution ──────────────────────────────────────────────────────────

function main() {
    // Check that an image is open
    if (ImageWindow.activeWindow.isNull) {
        (new MessageBox(
            "No image is currently open. Please open a linear FITS image first.",
            "AstroDeNoise", StdIcon_Error
        )).execute();
        return;
    }

    var activeWindow = ImageWindow.activeWindow;
    var view = activeWindow.mainView;

    // ── Linear data guard ──
    // Compute median of the image. Linear astrophotography data fresh from
    // calibration/integration typically has a median well below 0.05.
    // A stretched image will have a median of 0.3–0.6+.
    // This model was trained on linear data and WILL produce garbage on stretched input.
    var median = view.image.median();
    Console.writeln("AstroDeNoise: Image median = " + median.toFixed(5));

    if (median > 0.1) {
        var msg =
            "WARNING: This image appears to be non-linear (stretched).\n\n" +
            "Measured median pixel value: " + median.toFixed(4) + "\n" +
            "Expected median for linear data: < 0.05\n\n" +
            "AstroDeNoise was trained on linear (un-stretched) data.\n" +
            "Running it on stretched data will produce incorrect results.\n\n" +
            "Please use the original calibrated linear image as input.\n\n" +
            "Continue anyway?";
        var dlgWarn = new MessageBox(msg, "AstroDeNoise — Linear Data Warning",
                                     StdIcon_Warning, StdButton_Yes, StdButton_No);
        if (dlgWarn.execute() !== StdButton_Yes) {
            return;
        }
    }

    // Show dialog
    var dlg = new AstroDeNoiseDialog();
    if (dlg.execute() !== StdDialogCode_Ok) {
        return;
    }

    var modelPath = dlg.modelEdit.text.trim();
    var tileSize  = dlg.tileSizeSpinBox.value;

    if (modelPath === "" || !File.exists(modelPath)) {
        (new MessageBox(
            "Model file not found:\n" + modelPath,
            "AstroDeNoise", StdIcon_Error
        )).execute();
        return;
    }

    // ── Export current image to a temp FITS file ──
    var tmpDir = File.systemTempDirectory + "/astroenhance_";
    var inputTmp  = tmpDir + "input.xisf";
    var outputTmp = tmpDir + "output.xisf";

    Console.writeln("AstroDeNoise: Exporting image to temp file: " + inputTmp);

    // Save via PixInsight's XISF writer (native format — no precision loss)
    var fileFormat = new FileFormat("XISF", false, true);
    if (!fileFormat.isNull) {
        var fi = new FileFormatInstance(fileFormat);
        if (!fi.create(inputTmp, "")) {
            Console.criticalln("AstroDeNoise: Failed to create temp FITS file");
            return;
        }
        var desc = new ImageDescription();
        desc.bitsPerSample = 32;
        desc.ieeefpSampleFormat = true;
        if (!fi.setOptions(desc)) {
            Console.criticalln("AstroDeNoise: Failed to set FITS options");
            return;
        }
        if (!fi.writeImage(view.image)) {
            Console.criticalln("AstroDeNoise: Failed to write image data");
            return;
        }
        fi.close();
    } else {
        Console.criticalln("AstroDeNoise: FITS format not available");
        return;
    }

    // ── Call Python inference script ──
    var cmd = PYTHON_PATH +
        " \"" + INFER_SCRIPT + "\"" +
        " --model \"" + modelPath + "\"" +
        " --input \"" + inputTmp + "\"" +
        " --output \"" + outputTmp + "\"" +
        " --tile_size " + tileSize +
        " --overlap " + OVERLAP;

    Console.writeln("AstroDeNoise: Running Python inference...");
    Console.writeln("  Command: " + cmd);
    Console.flush();

    var exitCode = ExternalProcess.execute(cmd);

    if (exitCode !== 0) {
        (new MessageBox(
            "Python inference failed (exit code " + exitCode + ").\n\n" +
            "Check that:\n" +
            "  1. PYTHON_PATH is correct (" + PYTHON_PATH + ")\n" +
            "  2. Required packages are installed (torch, astropy)\n" +
            "  3. The model file is valid\n\n" +
            "See the PixInsight Process Console for details.",
            "AstroDeNoise Error", StdIcon_Error
        )).execute();
        return;
    }

    if (!File.exists(outputTmp)) {
        (new MessageBox(
            "Output file was not created. Check the Process Console log.",
            "AstroDeNoise Error", StdIcon_Error
        )).execute();
        return;
    }

    // ── Load the enhanced output back into PixInsight ──
    Console.writeln("AstroDeNoise: Loading enhanced image...");

    var loadOp = new ImageWindow(1, 1, 1, 32, true, false, "AstroDeNoise_result");
    var windows = ImageWindow.open(outputTmp);
    if (windows.length > 0) {
        windows[0].mainView.id = view.id + "_AstroDeNoised";
        windows[0].show();
        Console.writeln("AstroDeNoise: Done! Enhanced image opened as: " +
                        windows[0].mainView.id);
    } else {
        Console.criticalln("AstroDeNoise: Failed to load output FITS.");
    }

    // Cleanup temp files
    if (File.exists(inputTmp))  File.remove(inputTmp);
    if (File.exists(outputTmp)) File.remove(outputTmp);
}

main();
