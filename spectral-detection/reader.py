import logging
from pathlib import Path

import numpy as np
import os
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import _BAYER_CONFIG_RGGB, LORAWtoGRAY8, GRAY12toGRAY16
from lo.sdk.api.acquisition.io.open import open as loopen
from lo.sdk.api.camera.camera import LOCamera

import logging

try:
    import cupy

    xp = cupy
    logging.warning(
        "Cupy is successfully imported. "
        "GPU will be used during the execution of this program."
    )
except ImportError:
    logging.warning(
        "GPU enabled cupy failed to import, falling back to CPU. "
        "The execution may be slow."
    )
    import numpy

    xp = numpy


class LOReader:
    def __init__(
        self,
        calib_folder,
        source,
        calibration_frame=None,
        frame_rate=20,
        exposure=50,
        gain=0,
    ):
        """
        :param calib_folder:
        :param source:
        :param frame_rate:
        :param exposure:
        :param gain:
        """

        # Instantiate converters and calculators
        calibration_folder = Path(calib_folder).as_posix()
        self.decoder = SpectralDecoder.from_calibration(
            calibration_folder, calibration_frame
        )
        self.file_handler = None
        self.lo_format = False
        if source is not None:
            self.file_handler = loopen(source, "r")
            if os.path.splitext(source)[1] == ".lo":
                self.lo_format = True

        self.source = LOCamera(file=self.file_handler)

        # Open camera and turn on stream
        self.source.open()
        self.source.stream_on()

        if source is None:
            self.source.frame_rate = frame_rate * 1000 * 1000
            self.source.gain = gain
            self.source.exposure = exposure * 1000

    def __del__(self):
        self.source.stream_off()
        self.source.close()

        if self.file_handler is not None:
            self.file_handler.close()
            self.file_handler = None

    def __len__(self):
        if self.file_handler is not None:
            return len(self.file_handler)
        return -1

    def _debayer(self, frame, info, is_low_res=False):
        """
        Debayer low resolution
        :param frame:
        :param info:

        """
        out_dict: dict = {idx: [] for idx in range(3)}
        [
            out_dict[ci].append(
                (frame[bi] / bsi)[None, :, :, None].astype(frame[bi].dtype)
            )
            for bi, ci, bsi in zip(
                _BAYER_CONFIG_RGGB.bayer_index,
                _BAYER_CONFIG_RGGB.color_index,
                _BAYER_CONFIG_RGGB.fractions,
            )
        ]
        res = xp.concatenate(
            [xp.sum(xp.asarray(v), 0) for i, v in out_dict.items()], -1
        )

        if is_low_res:
            info.sampling_coordinates = xp.asarray(info.sampling_coordinates // 2)
        else:
            res = xp.repeat(xp.repeat(res, 2, axis=1), 2, axis=2)
        return xp.squeeze(res), info

    def get_next_frame(
        self, is_dynamic_range_normalize=False, is_low_res=False, return_raw=False
    ):
        """
        :param is_dynamic_range_normalize:
        :param is_low_res:
        """
        try:
            if self.file_handler is None:
                frame = self.source.get_frame()
            else:
                frame = self.file_handler.read()

            if self.lo_format:
                info, scene_frame, spectra = frame
            else:
                info, scene_frame, spectra = self.decoder(frame)

            if return_raw:
                return info, scene_frame, spectra

            if is_dynamic_range_normalize:
                scene_frame = xp.asarray(scene_frame)
                scene_frame = scene_frame - scene_frame.min()
                scene_frame = (
                    scene_frame / xp.percentile(scene_frame.flatten(), 97)
                ) * 255
                scene_frame = xp.clip(scene_frame, 0, 255)
                scene_frame = scene_frame.astype(xp.uint8)
            else:
                if self.lo_format:
                    scene_frame = (scene_frame * 2**8)/2**12
                else:
                    scene_frame = LORAWtoGRAY8(scene_frame)
            debayered, info = self._debayer(scene_frame, info, is_low_res)

            scene_frame = xp.asarray(debayered).astype(np.uint8)
            return info, scene_frame, spectra

        except Exception as e:
            logging.warning(f"Received exception {e}")
            return None, None, None

    def seek_frame(self, frame_idx):
        """
        :param frame_idx
        """
        if self.file_handler is None:
            logging.warning("Cannot seek frame index when running camera live.")
        else:
            self.file_handler.seek(frame_idx)

    def get_wavelengths(self):
        return self.decoder.calibration.wavelengths

    def get_sampling_coordinates(self):
        return self.decoder.sampling_coordinates
