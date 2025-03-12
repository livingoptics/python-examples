# ----------------------------- IMPORT PACKAGES ------------------------------
import os
import os.path as op
import cv2
import numpy as np
from lo.sdk.helpers.import_numpy_or_cupy import xp
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as lo_open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.api.acquisition.data.formats import _debayer

def percentile_norm(im: xp.ndarray, low: int = 1, high: int = 95) -> xp.ndarray:
    """
    Normalise the image based on percentile values.

    Args:
        im (xp.ndarray): The input image.
        low (int): The lower percentile for normalisation.
        high (int): The higher percentile for normalisation.

    Returns:
        xp.ndarray: The normalised image.
    """
    im[..., 0] = im[..., 0] - xp.percentile(im[::100, ::10, 0], low)
    im[..., 0] = im[..., 0] / xp.percentile(im[::100, ::10, 0], high)
    im[..., 1] = im[..., 1] - xp.percentile(im[::100, ::10, 1], low)
    im[..., 1] = im[..., 1] / xp.percentile(im[::100, ::10, 1], high)
    im[..., 2] = im[..., 2] - xp.percentile(im[::100, ::10, 2], low)
    im[..., 2] = im[..., 2] / xp.percentile(im[::100, ::10, 2], high)
    return xp.clip(im, 0, 1) * 255


# ------------------------------ DATA FILEPATHS ------------------------------
# Calibration location
factory_calibration_folder = "/datastore/lo/share/calibrations/latest_calibration"
# If you are running from workstation set this to None.

# Field calibration file
field_calibration_file = None

# File to load - pass None to stream directly from the camera
file = (
    "/datastore/lo/share/data/single-potato-peel-flesh-thin-peel-20240924-142815-386193.lo"
)
if file:
    lo_file = lo_open(file, "r")
    video_name = op.basename(file).split(".")[0]
else:
    lo_file = None
    video_name = "output"

# -------------------------- CREATE DECODER -----------------------------------
if factory_calibration_folder is not None:
    decoder = SpectralDecoder.from_calibration(factory_calibration_folder, field_calibration_file)


# ------------------------- Setup output location ----------------------------

# Output folder
output_folder = "./output"
video_folder = op.join(output_folder, "video")
spectra_folder = op.join(output_folder, "spectra")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)
os.makedirs(spectra_folder, exist_ok=True)

video_path = op.join(video_folder, video_name)

# Configure the video writer
fps = 10
shape = (2048, 2432, 96)
recorder = cv2.VideoWriter(
    video_path + ".mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (shape[1], shape[0]),
)

frame_number = 0
with LOCamera(file=lo_file) as cam:

    # When loading from a file, using the LOCamera method requires the sensor settings to be given. 
    cam.frame_rate = fps * 1e6
    cam.gain = 100
    cam.exposure = 633333

    while True:
        try:
            frame = cam.get_frame()
            if len(frame) == 4:
                metadata, scene, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)
            else:
                metadata, scene, spectra = frame
                scene = _debayer(np.squeeze(scene), metadata)

            # Colour correct scene view
            scene = percentile_norm(scene.astype(np.float32), high=99)

            # Write to file
            recorder.write(scene.astype(np.uint8))

        except KeyboardInterrupt:
            # Finalise mp4 video file
            recorder.release()
            break

        frame_number += 1