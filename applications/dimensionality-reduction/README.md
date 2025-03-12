# Dimensionality Reduction

This example shows how to perform dimensionality reduction using Living Optics hyperspectral
data. The following techniques are implemented:

1. Principle Component Analysis (PCA)
2. Minimum Noise Fraction (MNF)
3. Linear Discriminant Analysis (LDA)

## Preview

This preview shows how PCA can used to identify buising in apples.

<img src="./media/fruit-bruising.gif" height="300px"/>

## Prerequisites

- Have installed the Living Optics SDK as per the `Custom Python Environment` [install instructions](https://cloud.livingoptics.com/shared-resources?file=docs/ebooks/install-sdk.pdf).
- From within the SDK virtual environment, have installed these additional dependencies:

```bash
 pip install -r dr-requirements.txt
```

## Sample data

Sample data can be found
[here](https://cloud.livingoptics.com/shared-resources?file=samples/bruised-apple.zip)

## Usage

### Script

```bash
python dimensionality_reduction.py
```

### User inputs

- filepath - enter the path to the `.loraw` file
- factory_calibration_folder - enter the path to the factory calibration folder for your camera
- field_calibration_file - enter the path to the Field calibration file if required
- frame_idx - selects the frame from an `.loraw` file
- analysis_type - selects the analysis to run

Analysis is one of:

- "PCA" = Principle Component Analysis (PCA)
- "MNF" = Minimum Noise Fraction (MNF)
- "LDA" = Linear Discriminant Analysis (LDA)

### CLI usage

```bash
python dimensionality_reduction_cli.py --calibration-folder /path/to/factory/calibration/folder --filepath /path/to/loraw/file/data.loraw --calibration-frame-path /path/to/field-calibration --analysis-type 1
```

### Example PCA analysis of bruised apples:

```bash
python dimensionality_reduction_cli.py -c /datastore/lo/share/samples/bruised-apple/demo-calibration -f /datastore/lo/share/samples/bruised-apple/bruised-apple.loraw
```

### Example for PCA Analysis on frame 0

```bash
python dimensionality_reduction_cli.py --calibration-folder /datastore/lo/share/samples/bruised-apple/demo-calibration --filepath /datastore/lo/share/samples/bruised-apple/bruised-apple.loraw -at=PCA --frame-index 0
```

### Example for LDA Analysis on frame 3 in the data file

```bash
python dimensionality_reduction_cli.py -c /datastore/lo/share/samples/bruised-apple/demo-calibration -f /datastore/lo/share/samples/bruised-apple/bruised-apple.loraw -at LDA -i 3
```

### CLI tips

Once your device meets the software requirements -

- The `analysis-type` (_--analysis-type_ / _-at_) is where you need to specify which type of
  analysis you want to perform. (Mentioned in the **Introduction**)
- A window will pop up to for the user to select the Region of Interest (ROI) from the scene for
  White Reference and Background. Please select the brightest region for the White Reference and
  darkest for the Background using the mouse to draw ROI for each.
- **NOTE:** When you are performing LDA, another window will be shown to the user to select a region
  for the labels.
- Now, another window will pop up where you can see the result.