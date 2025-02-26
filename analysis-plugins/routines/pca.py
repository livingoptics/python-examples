# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import logging
from typing import Tuple

import numpy as np
from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis
from sklearn.decomposition import PCA

_logger = logging.getLogger(__name__)


class PrincipalComponentAnalysisExample(BaseAnalysis):
    def __init__(self, **kwargs):
        super(PrincipalComponentAnalysisExample, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(output_shape=(1920, 1920), origin=(64, 256))

        self.pca = None
        self.n_components = 96

    def _perform_pca(
        self,
        spectra
    ):
        self.pca = PCA(self.n_components)
        self.pca.fit(spectra)

    def _get_pca_dimension(self, spectra, pca_dimension_min, pca_dimension_max):
        
        self.transform = self.pca.transform(spectra)
        
        if pca_dimension_min < pca_dimension_max:
            pca_dimension_max = min(pca_dimension_max + 1, self.n_components)
            return self.transform[:, pca_dimension_min:(pca_dimension_max)].sum(axis=-1)
        if pca_dimension_min < self.n_components:
            return self.transform[:, pca_dimension_min]
        else:
            _logger.warning(
                "Invalid input for pca_dimensions...\nPCA Dimension value must not exceed n_components... Changing the value to 0"
            )
            return self.transform[:, 0]

    def __call__(
        self,
        loframe,
        spectrum_background: list,
        use_background: bool = False,
        n_components: int = 96,
        pca_dimension_min: int = 0,
        pca_dimension_max: int = 0,
        fit_pca: bool = True,
        truncate_outliers: bool = True,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Perform the Principle Component Analysis Clustering on a given frame
        Args:
            loframe (Tuple): Decoded frame information from camera
            (encoded_frame_info, encoded_frame), (scene_frame_info, scene_frame)
            spectrum_background (list): Spectrum of background region selected by user.
            use_background (optional, bool): Whether or not to use background.
            Defaults to False.
            pca_dimension_min (int): The lowest PCA component to return in the overlay.
            pca_dimension_max (int): The highest PCA component to return in the overlay, when pca_dimension_min != pca_dimension_max returns a sum of the components.
            fit_pca (bool): If True the PCA model is fitted to the incoming frame and then predictions are returned in the overlay. If False the last recorded PCA model is used to predict the current frame.
            truncate_outliers (bool) Whether outliers should be cropped
        """

        metadata, preview, spectra = loframe
        spectra = np.clip(spectra, np.finfo(np.float32).eps, None)

        self.n_components = n_components

        spectrum_background_np = np.clip(spectrum_background, np.finfo(np.float32).eps, None)
        if use_background:
            # convert to reflectance
            spectra = spectra / spectrum_background_np

        if fit_pca:
            self._perform_pca(spectra)
    
        pca = self._get_pca_dimension(spectra, pca_dimension_min, pca_dimension_max)
        
        # generate the simulated spectra
        spectra = self.pca.inverse_transform(self.transform)

        overlay: np.ndarray = self.upsampler(pca, sampling_coordinates=metadata.sampling_coordinates)

        if truncate_outliers:
            min, max = np.percentile(overlay, [3, 97])
            # clip outliers
            overlay = np.clip(overlay, min, max)

        padding = (
            (preview.shape[0] - overlay.shape[0]) // 2,
            (preview.shape[1] - overlay.shape[1]) // 2,
        )

        overlay = np.pad(overlay, ((padding[0], padding[0]), (padding[1], padding[1])))

        return (metadata, preview, spectra), overlay
