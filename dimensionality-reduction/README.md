# Dimensionality reduction

This shows how to perform dimensionality reduction using LO spatial spectral data. See support techinques below:

1. Principle Compenents Analysis (PCA)
2. Minimum Noise Fraction (MNF)
3. Linear Discriminant Analysis (LDA)

### Preview


Dimensionality reduction |
:------------: |
![Dimensionality reduction](./media/fruit-bruising.gif)  |


## Additional requirements

These examples require functions to perform the data analysis:

```bash
 pip install opencv-python
 pip install scikit-learn
 pip install matplotlib
 pip install pysptools
```

## Usage

### Script

```bash
python dimensionality_reduction/dimensionality_reduction.py 
```

### User inputs

- filepath - enter the path to the dataset for which the NDVI should be calculated
- calibration_folder - enter the path to the factory calibration folder for your camera
- calibration_frame_path - enter the path to the field calibration frame
- frame_idx - selects the frame from a video loraw file
- analysis_type - selects the analysis to run

Analysis is one of:

- "PCA" = Principle Compenents Analysis (PCA)
- "MNF" = Minimum Noise Fraction (MNF)
- "LDA" = Linear Discriminant Analysis (LDA)

### CLI usage

```bash
python dimensionality_reduction/dimensionality_reduction_cli.py --calibration path/to/calibration/folder --filepath /path/to/loraw/file/data.loraw --atype 1
```

### Example for Analysis 1 (PCA)

```bash
python dimensionality_reduction_cli.py --calibration /home/user/projects/lo-calibration-1001231132-35mm-5.6_0_0 --filepath /home/user/projects/bruised_apple_centred.loraw --atype 1 --instance 0
```

### Example for Analysis 3 (LDA) on 3rd frame in the data file

```bash
python dimensionality_reduction_cli.py -c /home/user/projects/lo-calibration-1001231132-35mm-5.6_0_0 -f /home/user/projects/bruised_apple_centred.loraw -at 3 -i 3
```

### CLI tips

Once your device meets the software requirements -

- The *--atype* / *-at* is where you need to specify which type of analysis you want to perform. (Mentioned in the **Introduction**)
- A window will pop up to for the user to select the Region of Interest (ROI) from the scene for White Reference and Background. Please select the brightest region for the White Reference and darkest for the Background using the mouse to draw ROI for each.
- **NOTE:** When you are performing LDA, another window will be shown to the user to select a region for the labels.
- Now, another window will pop up where you can see the result.