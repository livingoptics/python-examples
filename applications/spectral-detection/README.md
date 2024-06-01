# Object identification using spectral classification

This show an example of how a simple KNN classifier, can be used to identify objects and substances with Living Optics spatial spectral information.

## Preview

Spectral detection |
:------------: |
![Spectral detection](./media/liquid-classification.gif)  |

## Additional requirements

These examples require the following dependencies:

```bash
pip install -r requirements.txt
```

## Usage

This example some in two forms one being a scripted example, and the other is a plugin to the LO analysis tool. both require the Living Optics SDK to be installed.

### script

#### To train a classifier against lo data and segmentation mask.

```bash
python spectral-detection/train_classifier.py
```
This script runs using example data and generates a classifier plus associated metadata which can be used in the run classifier script. To extend this to additional lo data it requires a segmentation label mask to be generated and passed to the script. The mask file is expected to be a `.npy` format file containing a array with shape `C,Y,X` where C is the number of classes and Y,X is the shape of the scene view. To generate the mask it is recommended to use the scene view of the lo format file using a segmentation tool such as the ones described [here](https://foobar167.medium.com/open-source-free-software-for-image-segmentation-and-labeling-4b0332049878) 



#### Running inference with a trained classifier.
```bash
python spectral-detection/run_classifier.py
```

This script runs inference on a classifier generated using the training script and visualises the result.

### User inputs

The script will require updates to the following variables.

- filepath - enter the path to a lo format video.
- calibration_folder - enter the path to the living optics calibration folder.
- classifier_path - path to sklearn knn classifier.
- scaler_path - path to sklearn scaler transformation.
- masks - path to segmentation mask file see above for details.
- metadata_path - path to label metadata.

### Analysis tool plugin

To run the analysis tool plugin do the following steps:

Add directory path to PYTHONPATH

```bash
cd spectral-detection
export PYTHONPATH="${PYTHONPATH}:`pwd`"
```

Then, run the tool

```bash
analysis 
--analysis analysis_routine.DetectionAnalysis
--file  PATH_TO_LORAW_FILE
--calibration-path PATH_TO_LO_CALIBRATION
```