# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import numpy as np

from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.tools.analysis.apps import BaseAnalysis


class SingleBandAnalysis(BaseAnalysis):
    def __init__(self, **kwargs):
        super(SingleBandAnalysis, self).__init__(**kwargs)
        self.upsampler = NearestUpSample(output_shape=(1920, 1920), origin=(64, 256))

    def __call__(
        self, loframe, wavelength_min: float = 440, wavelength_max: float = 460, truncate_outliers: bool = True
    ):
        """Returns the sum of intensity for a given wavelength band
        Args:
            loframe (Tuple): Decoded frame information from camera
            (encoded_frame_info, encoded_frame), (scene_frame_info, scene_frame)
            wavelength_min (float): the lowest wavelength
            wavelength_max (float): The highest wavelength, When wavelength_min and wavelength_max are equal will return the nearest wavelength channel based on the channel midpoint wavelength.
            truncate_outliers (bool) Whether outliers should be cropped
        """
        metadata, preview, spectra = loframe

        start, stop = [np.argmin(np.abs(metadata.wavelengths - i)) for i in [wavelength_min, wavelength_max]]

        if start == stop:
            band = spectra[:, start]
        elif start < stop:
            band = spectra[:, start:stop].mean(axis=-1)
        else:
            band = spectra[:, stop:start].mean(axis=-1)

        if truncate_outliers:
            min, max = np.percentile(band, [3, 97])
            # clip outliers
            band = np.clip(band, min, max)

        map = self.upsampler(band, sampling_coordinates=metadata.sampling_coordinates)

        padding = (
            (preview.shape[0] - map.shape[0]) // 2,
            (preview.shape[1] - map.shape[1]) // 2,
        )
        map = np.pad(map, ((padding[0], padding[0]), (padding[1], padding[1])))
        return loframe, map
