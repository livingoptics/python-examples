# Enhancing Object detectoion with spectral classifcation:

These scripts show how to use [YOLO](https://github.com/ultralytics/ultralytics) object detectors with spectral information to enhance and filter detection results.

## Preview

Spectral enhanced object detection |
:------------: |
![Spectral enhanced object detection](./media/plastic-apple-id.gif)  |

## Prerequisites

- Have installed the Living Optics SDK as per the `Custom Python Environment` [install instructions](https://cloud.livingoptics.com/shared-resources?file=docs/ebooks/install-sdk.pdf).
- From within the SDK virtual environment, have installed these additional yolo dependencies:

```bash
pip install -r eod-requirements.txt
```

## Sample data

Sample data can be found
[here](https://cloud.livingoptics.com/shared-resources?file=samples/enhanced-object-detection.zip)

## Usage

### Canned Demo

YOLO detection using a recorded LO file:

```bash
python ./object_detection_with_spectral_filtering_file.py
```

To run the spectral filtering demo, click on the top-left apple in the frame. The selected apple should then be uniquely tracked across the subsequent frames in the video.

### User Inputs

- filepath - enter the path to the dataset for which the NDVI should be calculated
- factory_calibration_folder - enter the path to the factory calibration folder for your camera
- field_calibration_file - enter the path to the field calibration file

### Live Demo

YOLO object detection and spectral classification using a live camera feed:

```bash
python ./object_detection_with_spectral_filtering_live.py
```

Apple types are assigned by clicking on the YOLO-generated ROI.

### User Inputs

- factory_calibration_folder – Path to the factory calibration folder.
- field_calibration_file – Path to the field calibration frame.

### Parameters

- scale_factor - (int) this downscales the input frame passed to the huggingface model to improve speed.
- multi_class - whether to permit multiple classes or not.
- classification_threshold - the threshold above which a spectra will be considered part of the background class.
- binary_threshold - proportion of pixels in an ROI above/below the background threshold to consider the ROI as a True detection.
- storage_threshold - Minimum spectral angle to consider a classified spectrum as different from previously classified spectra, and, therefore, to assign a new object name.
- debug - Set to true to print the confidences and whether a sufficient number of points within each ROI were detected for a valid detection to occur. This is useful for figuring out the correct classification_threshold, storage threshold and binary threshold.