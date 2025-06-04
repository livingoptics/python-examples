# Jupyter notebook examples
This contains example notebooks for using with jupyter.

## Installation

### SDK

Visit [here](https://docs.livingoptics.com/sdk/install-guide.html) for instructions on how to install the SDK.

### Jupyter

For further detail on jupyter notebook installations, visit [jupyter website](https://jupyter.org/install).

```bash
SDK_PATH = 'PATH/TO/SDK'
source $SDK_PATH/bin/venv/bin/activate
pip install -r requirements.txt
```

## Examples and tutorials listing

Download sample data from the [Living Optics cloud portal](https://cloud.livingoptics.com/shared-resources?file=data/samples_v2/face-spoofing.zip).

| Example                                                      | Jupyter Notebook                                                   | data                                                                                      |
|:-------------------------------------------------------------|:-------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Stream from a Living Optics Camera                           | [Notebook](notebooks/stream.ipynb)                                 |                                                                                           |
| Stream to file decoded                                       | [Notebook](notebooks/stream_to_file_decoded.ipynb)                 |                                                                                           |
| Face detection using jetson-inference                        | [Notebook](notebooks/face_detection.cloud.livingoptics.com/shared-resources?file=data/samples_v2/livingoptics.com/shared-resources?file=data/samples_v2/face-spoofing.zip) |
| Face detection using jetson-inference and Spectral filtering | [Notebook](notebooks/face_detection_with_spectral_filtering.ipynb) | [data](https:cloud.livingoptics.com/shared-resources?file=data/samples_v2/2/face-spoofing.zip) |