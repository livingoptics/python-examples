# Object identification using spectral classification

This example shows how a simple KNN classifier can be used to identify different objects and substances with Living Optics hyperspectral data.

## Preview

Spectral detection |
:------------: |
![Spectral detection](./media/liquid-classification.gif)  |

## Prerequisites

- Have installed the Living Optics SDK as per the `Custom Python Environment` [install instructions](https://cloud.livingoptics.com/shared-resources?file=docs/ebooks/install-sdk.pdf).
- From within the SDK virtual environment, have installed these additional dependencies:

```bash
pip install -r sd-requirements.txt
```

## Sample data

Sample data can be found
[here](https://cloud.livingoptics.com/shared-resources?file=data/samples_v2/spectral-detection.zip).

## Usage

This example comes in two forms, both of which require the Living Optics SDK to be installed:
- a scripted example
- a plugin to the Living Optics analysis tool.

### User inputs

The scripts will require updates to the following variables.

- filepath - enter the path to an `.lo` format video.
- classifier_path - path to sklearn knn classifier.
- scaler_path - path to sklearn scaler transformation.
- masks - path to segmentation mask file.
- metadata_path - path to label metadata.

## Scripted version

### Train a classifier against `.lo` data and segmentation mask.

```bash
python ./train_classifier.py
```
This script runs using example data and generates a classifier plus associated metadata which can be used in the run classifier script. To extend this to additional `.lo` data it requires a segmentation label mask to be generated and passed to the script. The mask file is expected to be a `.npy` format file containing an array with shape `C,Y,X` where C is the number of classes and Y,X is the shape of the scene view. To generate the mask it is recommended to use the scene view from the `.lo` format file using a 3rd party segmentation tool such as the ones described [here](https://foobar167.medium.com/open-source-free-software-for-image-segmentation-and-labeling-4b0332049878) 

### Run inference with a trained classifier.

```bash
python ./run_classifier.py
```

This script runs inference on a classifier generated using the `train_classifier.py` script and visualises the result.

### Analysis tool plugin

To run the analysis tool plugin do the following steps:

- Add directory path to PYTHONPATH:

```bash
cd spectral-detection
export PYTHONPATH="${PYTHONPATH}:`pwd`"
```

- Then, run the tool:

```bash
analysis 
--file  PATH_TO_LO_FILE
```
