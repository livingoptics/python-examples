# Jetson inference integration

This contains integrations of popular jetson libaries with the Living Optics camera.

## Installation

### SDK

Visit [here](https://docs.livingoptics.com/sdk/install-guide.html) for instructions on how to install the SDK.

### jetson-inference and jetson-utils

Visit [here](https://github.com/dusty-nv/jetson-inference?tab=readme-ov-file#system-setup) for instructions on how to install jetson-inference which also installs jetson-utils as a submodule.

### Integrating Jetson libraries into the SDK venv

After installing jetson libraries, which installs it globally, run the following:

```sh
cd VENV_PATH/lib/python3.8/site-packages
ln -s /usr/lib/python3.8/dist-packages/jetson_utils_python.so .
ln -s /usr/lib/python3.8/dist-packages/jetson_utils .
ln -s /usr/lib/python3.8/dist-packages/jetson_inference_python.so .
ln -s /usr/lib/python3.8/dist-packages/jetson .
ln -s /usr/lib/python3.8/dist-packages/Jetson .
```

This will link symlinks to the jetson libraries to your python venv. Alternatively, you can create copies by replacing `ln -s` by `cp -r`.



## Examples

For more details on an example, click on any of the images or GIFs below. Full [listing](#examples-listing) below.

Stream from camera | Stream to file decoded |
:------------: |  :----------: |
[![Stream from camera](./media/face_spoof.gif)](./notebooks/stream.ipynb)  | [![Stream to file decoded](./media/scene_with_spectra.gif)](./notebooks/stream_to_file_decoded.ipynb)  |

Face detection | Face detection with Spectral filtering |
:------------: |  :----------: |
[![Face detection](./media/face_spoof_tracking.gif)](./notebooks/face_detection.ipynb)  | [![Face detection with Spectral filtering](./media/face_spoof_tracking_with_spectral_filtering.gif)](./notebooks/face_detection_with_spectral_filtering.ipynb)  |

## Examples listing

Example | Python script |
:--------------------|:-------|
Stream to file decoded | [Script](python/stream_to_file_decoded.py) |
Face detection using jetson-inference | [Script](python/face_detection.py) |
Face detection using jetson-inference and Spectral filtering | [Script](python/face_detection_with_spectral_filtering.py) |