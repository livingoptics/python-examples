# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os

from lo.sdk.api.acquisition.io.open import open as lo_open
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer

# Setup
#PATH_TO_LO_FILE = "/datastore/lo/share/samples_v2/face_spoofing/face_spoofing.lo"
PATH_TO_LO_FILE = "/Users/james/Downloads/face-spoofing/face-spoof-demo.lo"

# Get image and spectra viewers
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