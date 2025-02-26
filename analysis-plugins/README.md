# Sample Analysis Tool Routines

This section contains examples of analysis routines which are compatible with the PYQT based [analysis tool](https://docs.livingoptics.com/sdk/tools/analysis/tool-guide.html).


## Preview

| Script                                             | Functionality                                    |
| -------------------------------------------------- | ------------------------------------------------ |
| [Single band](./routines/single_band.py)           | ![Single band](./media/single_band.gif)          |
| [Band ratio](./routines/band_ratio.py)             | ![Band ratio](./media/band_ratio.gif)            |
| [Principal Component Analysis](./routines/pca.py)  | ![Principal Component Analysis](./media/pca.gif) |
| [K-means Clustering](./routines/kmeans.py)         | ![K-means Clustering](./media/kmeans.gif)        |
| [Mean-shift clustering](./routines/mean_shift.py)  | ![Mean-shift clustering](./media/mean_shift_clustering.png) |
| [Anomaly detection](./routines/rxd.py)             | ![Anomaly detection](./media/anomaly_detection.gif)        |
|[MCARI (Modified Chlorophyll Absorption in Reflectance Index)](./routines/mcari.py)| ![MCARI](./media/mcari.gif) | 

## Usage

### Installation

#### SDK
1. You'll need to be first registered/logged in with our cloud service through [here](https://cloud.livingoptics.com/login)
2. Then download the SDK Then visit [here](https://cloud.livingoptics.com/shared-resources?file=docs/ebooks/install-sdk.pdf) for instructions on how to install the SDK.

### To run

In a terminal window, activate the SDK virtual environment.
```bash
source bin activate venv
```

Add directory path to PYTHONPATH

```bash
export PYTHONPATH="${PYTHONPATH}:/PATH/TO/LOCATION/OF/ANALYSIS/ROUTINE"
```

Then, as in the documentation, do

```bash
analysis 
--analysis band_ratio.BandRatioAnalysis
--file /path/to/lo-file
--calibration-path /datastore/lo/share/calibrations/latest_calibration
```

## Example Scripts

- Single band
  - The intensity of a wavelength (nearest) or range of wavelengths.
  - Dataset example: `/datastore/lo/share/samples/ndvi/NDVI-demo.loraw`.

```bash
analysis --analysis single_band.SingleBandAnalysis --file /datastore/lo/share/samples/ndvi/NDVI-demo.loraw --calibration /datastore/lo/share/samples/ndvi/demo-calibration-ndvi
```

- Band ratio script
  - A band ratio is a simply a quotient of some select bands in a spectrum, which produces a single value for each spectral sample.
  - Different band ratios allows the extraction of different properties.
  - We have chosen the 2-band ratio here as an example, where the value is R = (B1-B2)/(B1+B2).
  - Dataset example can be downloaded from [here](https://cloud.livingoptics.com/shared-resources?file=samples/macbeth.zip).


```bash
analysis --analysis band_ratio.BandRatioAnalysisExample --file /datastore/lo/share/samples/bruised-apple/bruised-apple.loraw --calibration /datastore/lo/share/samples/bruised-apple/demo-calibration-bruised-apple
```

- Principal Component Analysis (PCA)
  - PCA is an algorithm for dimensionality reduction and is typically used for data preprocessing and exploration.
  - High dimensional spectral data is re-projected into a coordinate space where the first few components capture the direction of 'greatest variance' within the dataset
  - This analysis plugin allows you to display the channels one-by-one using an overlay on the image
  - Dataset example can be downloaded from [here](https://cloud.livingoptics.com/shared-resources?file=samples/macbeth.zip)

```bash
analysis --analysis pca.PrincipalComponentAnalysisExample --file /datastore/lo/share/macbeth/macbeth.loraw --calibration /datastore/lo/share/samples/ndvi/demo-calibration-macbeth
```

- K-means Clustering
  - K-means is an unsupervised learning method, where it assigns a each spectral datapoint to one of K classes
  - The user is able to choose the value of K. In this implementation it should be between 2-5.
  - The tool produces an overlay onto the image with the value of the label and the user can change the colour LUT by right-clicking on the colourbar to assign different colours to the classes.
  - Dataset example can be downloaded from [here](https://cloud.livingoptics.com/shared-resources?file=samples/macbeth.zip)


```bash
analysis --analysis kmeans.KMeansClustering --file /datastore/lo/share/samples/macbeth/macbeth.loraw  --calibration /datastore/lo/share/samples/ndvi/demo-calibration-macbeth
```

- Mean-shift Clustering
  - Mean shift clustering is another unsupervised learning algorithm. It is centroid based and the user does not need to choose the number of clusters.
  - We can apply it to spectra to automatically group together 'similar' points
  - Dataset example can be downloaded from [here](https://cloud.livingoptics.com/shared-resources?file=samples/macbeth.zip)

```bash
analysis --analysis mean_shift.MeanShiftClusterer --file /datastore/lo/share/samples/macbeth/macbeth.loraw  --calibration /datastore/lo/share/samples/ndvi/demo-calibration-macbeth
```

- Rx anomaly detection
  - The Red-Xiaoli detection script compares the statistics of each spectral point with the background (the average of the entire image).
  - Assuming that the anomalous points are rare compared to the background, they will be highlighted via an overlay.
  - Dataset example can be downloaded from [here](https://cloud.livingoptics.com/shared-resources?file=samples/anomaly-detection.zip)

```bash
analysis --analysis rxd.RxAnomalyDetector --file /datastore/lo/share/samples/anomaly_detection/anomaly-detection.loraw --calibration /datastore/lo/share/samples/anomaly-detection/demo-calibration-anomaly
```

- MCARI (Modified Chlorophyll Absorption in Reflectance Index)
  - MCARI (Modified Chlorophyll Absorption in Reflectance Index) is calculated using green, red and near-infrared (NIR) wavelengths.
  - Is used as a plant health measure and correlates well to chlorophyll content of leafs.
  - Dataset example can be downloaded from [here](https://cloud.livingoptics.com/shared-resources?file=samples/tree-with-blossoms.zip)

```bash
analysis --analysis mcari.MCARIAnalysis --file /datastore/lo/share/samples/tree-with-blossoms/tree-with-blossoms.loraw --calibration /datastore/lo/share/samples/tree-with-blossoms/demo-calibration-blossoms
```