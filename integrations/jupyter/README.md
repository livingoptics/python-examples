# Jupyter notebook examples
This contains example notebooks for using with jupyter.

## Installation

### SDK

Visit [here](https://docs.livingoptics.com/sdk/install-guide.html) for instructions on how to install the SDK.

### Jupyter

For further detail on jupyter notebook installations visit [jupyter website](https://jupyter.org/install)

```bash
SDK_PATH = 'PATH/TO/SDK'
source $SDK_PATH/bin/venv/bin/activate
pip install -r requirements.txt
```

## Examples and tutorials listing

To retrieve some sample data to explore navigate to the [LO cloud](https://docs.livingoptics.com/install-guide.html) and download the sample sample which interests you.

Example | Jupyter Notebook | data|
:--------------------|:--------------------------------------|-------|
Stream from LO Camera | [Notebook](notebooks/stream.ipynb) | 
Stream to file decoded | [Notebook](notebooks/stream_to_file_decoded.ipynb) | 
Face detection using jetson-inference | [Notebook](notebooks/face_detection.ipynb) |  [data](https://cloud.livingoptics.com/shared-resources?file=samples/spectral-detection.zip)
Face detection using jetson-inference and Spectral filtering | [Notebook](notebooks/face_detection_with_spectral_filtering.ipynb) | [data](https://cloud.livingoptics.com/shared-resources?file=samples/spectral-detection.zip)