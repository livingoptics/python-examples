# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import logging
from typing import Tuple

import numpy as np
from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis
from sklearn.cluster import KMeans

_logger = logging.getLogger(__name__)


class KMeansClustering(BaseAnalysis):
    def __init__(self, **kwargs):
        super(KMeansClustering, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(
            output_shape=(1920, 1920),
            origin=(64, 256),
        )

    def _perform_kmeans(
        self,
        spectra,
        clusters,
    ):
        if clusters > 5 or clusters < 0:
            _logger.warning(
                "Invalid input for clusters...\nSetting defualt value of 2..."
            )
            clusters = 2

        labels = (
            KMeans(n_clusters=clusters, random_state=0, n_init="auto")
            .fit(spectra)
            .labels_
        )

        labels = labels * (255 // (clusters - 1))

        return labels.astype(np.int16)

    def __call__(
        self,
        loframe,
        spectrum_background: list,
        use_background: bool = False,
        clusters: int = 2,
    ):
        """Perform the KMeans Clustering on a given frame
        (overlays the clustered sections)
        Args:
            loframe (Tuple): Decoded frame information from camera
            (encoded_frame_info, encoded_frame), (scene_frame_info, scene_frame)
            spectrum_background (list): Spectrum of background region selected by user.
            use_background (optional, bool): Whether or not to use background.
            Defaults to False.
            clusters (optional, int): Number of clusters.
            Values should be between 0-5. Defaults to 2.
        """
        metadata, preview, spectra = loframe
        spectra = np.clip(spectra, np.finfo(np.float32).eps, None)
        spectrum_background_np = np.clip(
            spectrum_background, np.finfo(np.float32).eps, None
        )
        if not use_background:
            spectrum_background_np[:] = 1

        reflectance = spectra / spectrum_background_np

        labels = self._perform_kmeans(reflectance, clusters)
        overlay: np.ndarray = self.upsampler(labels, metadata.sampling_coordinates)

        padding = (
            (preview.shape[0] - overlay.shape[0]) // 2,
            (preview.shape[1] - overlay.shape[1]) // 2,
        )
        overlay = np.pad(overlay, ((padding[0], padding[0]), (padding[1], padding[1])))

        return loframe, overlay
