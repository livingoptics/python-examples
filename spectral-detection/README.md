# Object identification using spectral classification

This show an example of how a simple KNN classifier, can be used to identify objects and substances with Living Optics spatial spectral information.

## Preview

Spectral detection |
:------------: |
![Spectral detection](./media/liquid-classification.gif)  |

## Additional requirements

These examples require the following dependencies:

```bash
pip install scikit-learn
```

## Usage

This example some in two forms one being a scripted example, and the other is a plugin to the LO analysis tool. both require the Living Optics SDK to be installed.

### script

```bash
python spectral-detection/run_classifier.py
```

### User inputs

The script will require updates to the following variables.

- filepath - enter the path to a lo format video.
- calibration_folder - enter the path to the living optics calibration folder.
- classifier_path - path to sklearn knn classifier.
- scaler_path - path to sklearn scaler transformation.
- metadata_path - path to label metadata.

### Analysis tool plugin

To run the analysis tool plugin do the following steps:

Add directory path to PYTHONPATH

```bash
export PYTHONPATH="${PYTHONPATH}:spectral-detection/analysis_routine.py"
```

Then, run the tool

```bash
analysis 
--analysis analysis_routine.DetectionAnalysis
--file  PATH_TO_LORAW_FILE
--calibration-path PATH_TO_LO_CALIBRATION
```
