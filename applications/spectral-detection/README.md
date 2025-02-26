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

## Sample data

Sample data can be found
[here](https://cloud.livingoptics.com/shared-resources?file=samples/spectral-detection.zip)
```

## Usage

This example comes in two forms: one is a scripted example, and the other is a plugin for the LO analysis tool. Both require the Living Optics SDK to be installed.

### script

#### To train a classifier against lo data and segmentation mask.

```bash
python spectral-detection/train_classifier.py
```
This script runs using example data and generates a classifier along with associated metadata, which can be used in the run classifier script. To extend this to additional LO data, a segmentation label mask must be generated and passed to the script. The mask file should be in `.npy` format and contain an array with the shape `C, Y, X`, where C is the number of classes, and (Y, X) represents the shape of the scene view. To generate the mask, it is recommended to use the scene view of the LO format file with a segmentation tool, such as the ones described [here](https://foobar167.medium.com/open-source-free-software-for-image-segmentation-and-labeling-4b0332049878).


#### Running inference with a trained classifier.
```bash
python spectral-detection/run_classifier.py
```

This script runs inference on a classifier generated using the training script and visualises the result.

### User inputs

The script will require updates to the following variables.

- filepath - enter the path to a lo format video.
- classifier_path - path to sklearn knn classifier.
- scaler_path - path to sklearn scaler transformation.
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
--file  PATH_TO_LORAW_FILE
--calibration-path PATH_TO_LO_CALIBRATION
```

Please see the [Getting started guide](https://developer.livingoptics.com/getting-started/) for other options.