# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import numpy as np

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis
from sklearn.cluster import MeanShift


class MeanShiftClusterer(BaseAnalysis):

    def __init__(self, **kwargs):
        super(MeanShiftClusterer, self).__init__(**kwargs)
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

        rgb_map = MeanShift().fit_predict(spectra)
        overlay = self.upsampler(rgb_map, metadata.sampling_coordinates)

        padding = (
            (preview.shape[0] - overlay.shape[0]) // 2,
            (preview.shape[1] - overlay.shape[1]) // 2,
        )
        overlay = np.pad(overlay, ((padding[0], padding[0]), (padding[1], padding[1])))

        return loframe, overlay
