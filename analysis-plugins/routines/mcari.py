# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.
import math
import numpy as np
from lo.sdk.api.acquisition.data import Calibration
from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps.spectral_decode import SpectralDecode
from lo.sdk.tools.webcamera.utils.color_map import get_linear_cts_legend
from lo.sdk.tools.analysis.apps import BaseAnalysis

def MCARI(
    spectral_list: np.ndarray,
    wavelengths: np.ndarray,
    green_limits=(545, 555),
    red_limits=(668, 675),
    nir_limits=(695, 703),
):
    """
    Compute MCARI overlay from a spectral list and upsample it to match scene coordinates.

    MCARI (Modified Chlorophyll Absorption in Reflectance Index) is calculated using green, red and near-infrared (NIR) wavelengths.
    This function calculates MCARI for each point in the spectral list.

    Args:
        spectral_list (np.ndarray): Array of spectral data.
        wavelengths (np.ndarray): Array of wavelengths corresponding to the spectral data.
        green_limits (tuple, optional): Half open wavelength limits for green light. Defaults to [6540, 560).
        red_limits (tuple, optional): Half open wavelength limits for green light. Defaults to [660, 680).
        nir_limits (tuple, optional): Half open wavelength limits for near-infrared light. Defaults to [700, 720).
    Returns:
        ndvi (np.ndarray): MCARI calculated from the spectral list and wavelengths.
    Example:
        ```python
            info, scene, spectra = lo_frame  # Assume lo_frame is a predefined tuple
            spectral_list = spectra
            wavelengths = info.wavelengths
            mcari = MCARI(spectral_list, wavelengths)
            mcari.shape
        ```
    """
    start, stop = [np.argmin(np.abs(wavelengths - i)) for i in green_limits]
    green = spectral_list[:, start:stop].mean(axis=-1)

    start, stop = [np.argmin(np.abs(wavelengths - i)) for i in red_limits]
    red = spectral_list[:, start:stop].mean(axis=-1)

    start, stop = [np.argmin(np.abs(wavelengths - i)) for i in nir_limits]
    nir = spectral_list[:, start:stop].mean(axis=-1)

    # Handle division by zero
    red = np.where(red == 0, 1e-10, red)
    
    return ((nir - red) - 0.2 * (nir - green)) * (nir / red)



class MCARIAnalysis(BaseAnalysis):
    def __init__(self, **kwargs):
        super(MCARIAnalysis, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(output_shape=(1920, 1920), origin=(64, 256))
        
    def __call__(self, loframe, spectrum_background: list, use_background: bool=False, **kwargs) -> tuple:
        """Performs MCARI (Modified Chlorophyll Absorption in Reflectance Index) on a given frame
        Args:
            loframe (tuple): Decoded frame information from camera
            spectrum_background
            use_background
        """
        metadata, preview, spectra = loframe
        if use_background:
            spectrum_background = np.array(spectrum_background)
            r_spectra = spectra / spectrum_background
            r_spectra = np.clip(r_spectra, 0 ,1.0)
        else:
            r_spectra = spectra
        mcari = MCARI(r_spectra, metadata.wavelengths)

        map = self.upsampler(mcari, metadata.sampling_coordinates)

        padding = (
            (preview.shape[0] - map.shape[0]) // 2,
            (preview.shape[1] - map.shape[1]) // 2,
        )
        mcari_map = np.pad(map, ((padding[0], padding[0]), (padding[1], padding[1])))
        return None, mcari_map, get_linear_cts_legend(-math.inf, math.inf)