# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os

from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.io.open import open as lo_open
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer
from lo.sdk.helpers.path import getdatastorepath


# Setup


factory_calibration_folder_DIR = os.path.join(getdatastorepath(), "lo", "share", "samples", "face-spoofing", "demo-calibration-face-spoofing")
PATH_TO_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "temp", "stream_to_file.lo")
os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)

NO_FRAMES_TO_CAPTURE = 10

# Instantiate the LO's spectral decoder
decoder = SpectralDecoder.from_calibration(factory_calibration_folder_DIR)


# We instantiate a decoder object using the `SpectralDecoder` api with the calibration folder for your LO camera.


with LOCamera() as cam, lo_open(PATH_TO_OUTPUT_FILE, "w", format="lo") as f:
    for i, frame in enumerate(cam):
        if i >= NO_FRAMES_TO_CAPTURE:
            break
        
        # Decode spectra and scene
        processed_frame = decoder(
            frame,
            scene_decoder=LORAWtoRGB8,
            description="Description"
        )

        # Write to output lo file
        f.write(processed_frame)


# We'll decode every frame from the camera using our instantiated decoder together with the scene_decoder which converts the scene from raw to RGB8 format.
# Then, we'll write the processed_frame into the output `.lo` file using LO's IO tool.

# Get LO's image and spectra viewers
viewer = LOMPLViewer()
scene_view = viewer.add_scene_view(title="Scene view")
spectra_view = viewer.add_spectra_view(title="Sample spectra")

with lo_open(PATH_TO_OUTPUT_FILE, "r") as f:
    for (metadata, scene, spectra) in f:
        # Add scene to the viewer
        scene_view.update(scene)
        # Add some sample spectra to the viewer
        spectra_view.update(
            spectra=spectra[::500, :],
            wavelengths=metadata.wavelengths
        )

        # Render the viewer
        viewer.render()