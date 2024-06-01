# Dimensionality Reduction

This example shows how to perform dimensionality reduction using Living Optics spatial spectral
data. The following techniques are implemented:

1. Principle Compenents Analysis (PCA)
2. Minimum Noise Fraction (MNF)
3. Linear Discriminant Analysis (LDA)

### Preview

This preview shows how PCA can used to identify buising in apples.

<img src="./media/fruit-bruising.gif" height="300px"/>

## Additional requirements

This examples require functions to perform the data analysis:

```bash
 pip install opencv-python
 pip install scikit-learn
 pip install matplotlib
 pip install pysptools
```

## Sample data

Sample data can be found
[here](https://cloud.livingoptics.com/shared-resources?file=samples/bruised-apple.zip)

## Usage

### Script

```bash
python dimensionality_reduction/dimensionality_reduction.py
```

### User inputs

- filepath - enter the path to the .loraw dataset
- calibration_folder - enter the path to the factory calibration folder for your camera
- calibration_frame_path - enter the path to the field calibration frame if required
- frame_idx - selects the frame from a video loraw file
- analysis_type - selects the analysis to run

Analysis is one of:

- "PCA" = Principle Compenents Analysis (PCA)
- "MNF" = Minimum Noise Fraction (MNF)
- "LDA" = Linear Discriminant Analysis (LDA)

### CLI usage

```bash
python dimensionality_reduction/dimensionality_reduction_cli.py --calibration-folder /path/to/calibration/folder --filepath /path/to/loraw/file/data.loraw --calibration-frame-path /path/to/field-calibration --analysis-type 1
```

### Example PCA analysis of bruised apples:

```bash
python dimensionality_reduction_cli.py -c /datastore/lo/share/samples/bruised-apple/demo-calibration -f /datastore/lo/share/samples/bruised-apple/bruised-apple.loraw
```

### Example for Analysis 1 (PCA)

```bash
python dimensionality_reduction_cli.py --calibration-folder /datastore/lo/share/samples/bruised-apple/demo-calibration --filepath /datastore/lo/share/samples/bruised-apple/bruised-apple.loraw -at=PCA --frame-index 0
```

### Example for Analysis 3 (LDA) on 3rd frame in the data file

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