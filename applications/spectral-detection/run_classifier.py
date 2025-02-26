# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import pickle
import time

import cv2
import numpy as np
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as sdk_open
from lo.sdk.helpers.import_numpy_or_cupy import xp
from matplotlib import cm
from sklearn.neighbors import KDTree


def spectral_angle_nd_to_vec(spectral_list, reference_spectrum):
    """Calculates the spectral angle difference in radians between each row
        of the spectral list and the reference spectrum.

    Args:
        spectral_list (np.ndarray): shape (N_spectra, N_channels)
        reference_spectrum (np.ndarray): shape (N_channels)

    Returns:
        list of SAM scores (np.ndarray): in radians shape (N_spectra)
    """
    return xp.arccos(
        xp.clip(
            xp.dot(spectral_list, reference_spectrum)
            / xp.linalg.norm(spectral_list, axis=1)
            / xp.linalg.norm(reference_spectrum),
            0,
            1,
        )
    )


# ------------------------------ USER PARAMETERS ------------------------------
# Define paths
scaler_path = "/datastore/lo/share/samples/spectral-detection/scaler.model"
classifier_path = "/datastore/lo/share/samples/spectral-detection/classifier.model"
metadata_path = "/datastore/lo/share/samples/spectral-detection/metadata.txt"

# file path is an .lo file.
file_path = "/datastore/lo/share/samples/spectral-detection/liquid-segmentation.lo"

#
debayer = True

# ---------------------------- LOAD TRAINED MODELS ------------------------------
scaler = pickle.load(open(scaler_path, "rb"))
classifier = pickle.load(open(classifier_path, "rb"))
metadata = pickle.load(open(metadata_path, "rb"))

white_background = None
if "reference" in list(metadata.keys()):
    white_background = np.array(metadata.pop("reference"))


# ---------------------------- VISUAL PARAMETERS ---------------------------------
colours = cm.get_cmap("tab10", len(metadata)).colors * 255
colours = [tuple([int(k) for k in item]) for item in colours]

bg_colour = (10, 10, 10)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 3
text_width = (
    max(
        [
            cv2.getTextSize(v[0], font, font_scale, thickness)[0][0]
            for k, v in metadata.items()
        ]
    )
    + 60
)
text_height = 40 * len(metadata) + 10


cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)

# ------------------------- ITTERATE BY FRAME ---------------------------------
with sdk_open(file_path) as f:
    # Iterate over frames
    for frame in f:
        info, scene_view, spectra = frame
        sampling_coordinates = info.sampling_coordinates.astype(np.int32)

        # Convert to reflectance using the white background used in model training
        if white_background is not None:
            spectra = spectra / white_background
        stime = time.time()

        # ------------------------- CREATE PREDICTIONS ---------------------------------
        # The trained model is subsampled for speed
        spectra = spectra[:, ::4]
        scaled_spectra = scaler.transform(spectra)
        # Subtract 1 from labels to match metadata indices
        labels = classifier.predict(scaled_spectra) - 1
        print(f"prediction time: {time.time() - stime}")

        # ------------------ PREPARE SCENE VIEW FOR DISPLAY ----------------------------
        shp = np.shape(scene_view)

        if shp[-1] != 3:  # If not already 3-channel RGB
            if np.max(scene_view) > 2**12:
                scene_view = scene_view >> 4
            if np.max(scene_view) > 2**8:
                fraction = 2**12 // 2**8
                scene_view = scene_view / fraction

            if debayer:
                scene_view = LORAWtoRGB8(np.squeeze(scene_view))

        scene_view = cv2.normalize(
            scene_view, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        scene_view = cv2.cvtColor(scene_view, cv2.COLOR_BGR2RGB)

        # ---------------- OVERLAY PREDICTIONS ON SCENE VIEW ---------------------------
        # Filter & Visualise predictions
        mask_view = np.zeros_like(scene_view)

        for k, v in metadata.items():
            # Filter false positives using spectral angle
            thr = v[2] * 2.5
            seg_mask = spectral_angle_nd_to_vec(scaled_spectra[labels == k], v[1]) < thr
            locs = sampling_coordinates[labels == k][seg_mask]

            # Filter false positives spatially
            tree = KDTree(locs)
            max_dist = tree.query_radius(locs, 50)
            max_neighbours = max([len(dist) for dist in max_dist])

            spatial_mask = np.asarray(
                [len(loc) > (max_neighbours * 0.5) for loc in max_dist]
            )
            locs = locs[spatial_mask]

            for pts in locs[:, ::-1]:
                cv2.circle(mask_view, pts, 15, colours[k], -1)

        # Draw legend
        mask_view = cv2.rectangle(
            mask_view, (10, 10), (text_width, text_height), bg_colour, -1
        )
        for k, v in metadata.items():
            mask_view = cv2.rectangle(
                mask_view, (20, 20 + 40 * k), (40, 40 + 40 * k), colours[k], -1
            )
            mask_view = cv2.putText(
                mask_view,
                v[0],
                (50, 40 + 40 * k),
                font,
                font_scale,
                colours[k],
                thickness,
            )

        scene_view[mask_view != 0] = cv2.addWeighted(
            scene_view[mask_view != 0], 0.3, mask_view[mask_view != 0], 0.7, 1
        )[:, 0]

        # Show
        cv2.imshow("Inference", scene_view[:, :, [2, 1, 0]] / 255)
        cv2.waitKey(2)
