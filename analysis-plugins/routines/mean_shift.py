# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import numpy as np
from sklearn.cluster import MeanShift

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis


class MeanShiftClustering(BaseAnalysis):
    def __init__(self, **kwargs):
        super(MeanShiftClustering, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(
            output_shape=(1920, 1920),
            origin=(64, 256),
        )
    def __call__(self, loframe):
        """Performs Mean Shift Clustering on a given frame
        (overlays the clustered sections)
        Args:
            loframe (tuple): Decoded frame information from camera
        """
        metadata, preview, spectra = loframe

        # Perform Mean Shift clustering
        print("Fitting/Predicting in progress")
        labels = MeanShift().fit_predict(spectra)
        print("Done predicting")
        
        # Generate RGB map for visualization
        n_clusters = len(np.unique(labels))  # Number of unique clusters

        # Upsample the clustered map to match the size of the preview
        map = self.upsampler(labels, metadata.sampling_coordinates)

        # Padding to ensure the map matches the size of the preview
        padding = (
            (preview.shape[0] - map.shape[0]) // 2,
            (preview.shape[1] - map.shape[1]) // 2,
        )
        mean_shift_map = np.pad(map, ((padding[0], padding[0]), (padding[1], padding[1])))

        # Generate categories for the legend
        categories = [{"label": f"Cluster {i+1}", "color": f"#{hex(255 - (i * 30))[2:]}00{i*40:02x}"} for i in range(n_clusters)]

        # Generate meta info (labels and colors)
        metadata = {
            "legend": {
                "type": "categories",
                "labels": categories,
            }
        }

        return None, mean_shift_map, metadata
