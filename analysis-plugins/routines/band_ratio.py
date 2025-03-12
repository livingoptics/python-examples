# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

# !!!!!!  WARNING   !!!!!! 
# The camera has fixed channel numbers and is therefore unable to differentiate arbitrary wavelengths to use as bands, despite the software plugin allowing user input like this. 
# The software will display the closest channel to the chosen wavelength ONLY.


import numpy as np

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis


def band_ratio(spectra, wavelengths, band_one=(650, 680), band_two=(785, 900)):
    start, stop = [np.argmin(np.abs(wavelengths - i)) for i in band_one]
    vis = spectra[:, start:stop].mean(axis=-1)
    start, stop = [np.argmin(np.abs(wavelengths - i)) for i in band_two]
    nir = spectra[:, start:stop].mean(axis=-1)
    return (nir - vis) / (nir + vis)


class BandRatioAnalysisExample(BaseAnalysis):
    def __init__(self, **kwargs):
        super(BandRatioAnalysisExample, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(output_shape=(1920, 1920), origin=(64, 256))

    def __call__(
        self,
        loframe,
        band_one_min: float = 400,
        band_one_max: float = 700,
        band_two_min: float = 700,
        band_two_max: float = 900
    ):
        """Returns The band ratio between two wavelength bands. This is a generalisation of NDVI to arbitrary wavelength bands.
        Args:
            loframe (Tuple): Decoded frame information from camera
            (encoded_frame_info, encoded_frame), (scene_frame_info, scene_frame)
            band_one_min (float)
            band_one_max (float)
            band_two_min (float)
            band_two_max (float)
            truncate_outliers (bool) Whether outliers should be cropped
        """
        metadata, preview, spectra = loframe

        ratio = band_ratio(spectra, metadata.wavelengths, band_one=[band_one_min, band_one_max], band_two=[band_two_min, band_two_max])
        upsampled = self.upsampler(ratio, metadata.sampling_coordinates)

        padding = (
            (preview.shape[0] - upsampled.shape[0]) // 2,
            (preview.shape[1] - upsampled.shape[1]) // 2,
        )
        upsampled = np.pad(upsampled, ((padding[0], padding[0]), (padding[1], padding[1])))
        return loframe, upsampled