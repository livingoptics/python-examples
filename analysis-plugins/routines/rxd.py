# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import numpy as np

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis

def calculate_rxd(spectra):
    """
    The Reed-Xiaoli Detector (RXD) algorithm extracts targets that are spectrally distinct from the image background. For RXD to be effective, the anomalous targets must be sufficiently small, relative to the background. Results from RXD analysis are unambiguous and have proven very effective in detecting subtle spectral features.

    Args:
        spectra (np.ndarray): A n*spectra array of spectral signatures
    """
    # Calculate mean and find each pixel's difference w.r.t. mean spectra
    mean_spectra = np.mean(spectra, axis=0, keepdims=True)
    mean_normalised_spectra = spectra - mean_spectra

    # Compute the inverse covariance matrix
    inv_cov_matrix = np.linalg.inv(np.cov(mean_normalised_spectra.T, ddof=0))

    rx_res = np.einsum(
        "jk,km,jm->j",
        mean_normalised_spectra,
        inv_cov_matrix,
        mean_normalised_spectra,
        optimize=True,
    )
    rx_res = np.sqrt(rx_res).astype(np.float32)
    return rx_res


class RxAnomalyDetector(BaseAnalysis):

    def __init__(self, **kwargs):
        super(RxAnomalyDetector, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(output_shape=(1920, 1920), origin=(64, 256))

    def __call__(self, loframe):
        """Performs the Reed-Xiaoli Detector (RXD) algorithm identifying targets that are spectrally distinct from the image background
        Args:
            loframe (Tuple): Decoded frame information from camera
        """
        metadata, preview, spectra = loframe

        rgb_map = calculate_rxd(spectra)
        overlay = self.upsampler(rgb_map, sampling_coordinates=metadata.sampling_coordinates)

        padding = (
            (preview.shape[0] - overlay.shape[0]) // 2,
            (preview.shape[1] - overlay.shape[1]) // 2,
        )
        overlay = np.pad(overlay, ((padding[0], padding[0]), (padding[1], padding[1])))
        overlay = np.float32(
            ((overlay - overlay.min()) / (overlay.max() - overlay.min())) * 1
        )
        return loframe, overlay
