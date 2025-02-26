# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os
import pickle

import cv2
import numpy as np
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as sdk_open
from lo.sdk.helpers.import_numpy_or_cupy import xp
from matplotlib import colormaps
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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
file_path = "/datastore/lo/share/samples/spectral-detection/liquid-segmentation.lo"

# ------------------------------ LOAD MASKS ------------------------------------
# mask_file = "/datastore/lo/share/samples/spectral-detection/segmentation-mask.npy"
# masks = np.load(mask_file)

masks = np.zeros((23, 2048, 2432))
masks[0:50, 0:500, 0] = 1
masks[1500:2000, 1500:2000, 1] = 1

# To extend this example to additional lo data a segmentation labeled mask required, in npy format.
# Mask is expected to be a a binary mask of C,Y,X where C is the number of classes.


# ------------------------------ INIT CLASSIFIER ------------------------------
scaler = StandardScaler()
classifier = KNeighborsClassifier(
    n_neighbors=5, weights="distance", n_jobs=len(os.sched_getaffinity(0)) - 1
)
colours = colormaps.get_cmap("tab10").colors * 255
colours = [tuple([int(k) for k in item]) for item in colours]

debayer = True


# Get first frame and it's spectra

# ------------------------------ READ LO FORMAT FILE ---------------------------
with sdk_open(file_path) as f:
    # Iterate over frames

    info, scene_view, spectra = next(f)  # Get the first frame in the file for fitting

    shp = np.shape(scene_view)
    if shp[-1] != 3:
        if np.max(scene_view) > 2**12:
            scene_view = scene_view >> 4
        if np.max(scene_view) > 2**8:
            fraction = 2**12 // 2**8
            scene_view = scene_view / fraction

        # Optional debayering
        if debayer:
            scene_view = LORAWtoRGB8(np.squeeze(scene_view))

    scene_view = cv2.normalize(
        scene_view, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    scene_view = cv2.cvtColor(scene_view, cv2.COLOR_BGR2RGB)

# The trained model is subsampled for speed
spectra = spectra[:, ::4]


spectra_mask = np.zeros(scene_view.shape[:2]).astype(bool)
sc = info.sampling_coordinates.astype(np.int32)

if type(sc) is not np.ndarray:
    # Cupy is in place, convert to numpy
    sc = sc.get()
    scene_frame = scene_view.get()

spectra_mask[sc[:, 0], sc[:, 1]] = True

# ------------------ PREPARE INDICES ---------------------------------------------
spectra_indices = [v[sc[:, 0], sc[:, 1]] * (k + 1) for k, v in enumerate(masks)]
bg_indices = np.invert(np.any(spectra_indices, axis=0))

# ------------------ PREPARE SPECTRA FROM CLASSES --------------------------------
bg_spectra = spectra[bg_indices]
fg_spectra = [spectra[i != 0] for i in spectra_indices]

# ------------------ PREPARE LABELS ----------------------------------------------
bg_labels = np.zeros(len(bg_spectra), dtype=np.int32)
# fg_labels = [np.zeros(len(s), dtype=np.int32) + np.unique(spectra_indices[i])[1]
#              for i, s in enumerate(fg_spectra)]
fg_labels = []
for i, s in enumerate(fg_spectra):
    unique_values = np.unique(spectra_indices[i])
    if len(unique_values) > 1:  # Check if there are at least two unique values
        fg_labels.append(np.zeros(len(s), dtype=np.int32) + unique_values[1])
    else:
        # Handle the case where there's only one unique value
        fg_labels.append(
            np.zeros(len(s), dtype=np.int32)
        )  # Assign default label (e.g., 0)

# ------------------ MERGE SPECTRA AND LABELS ------------------------------------
all_data = np.concatenate((bg_spectra, *fg_spectra), axis=0)
all_labels = np.concatenate((bg_labels, *fg_labels), axis=0)

# ------------------ FIT CLASSIFIER ----------------------------------------------
all_data_scaled = scaler.fit_transform(all_data)
classifier.fit(all_data_scaled, all_labels)

# ------------------ SAVE CLASSIFIER ---------------------------------------------
pickle.dump(scaler, open("scaler.model", "wb"))
pickle.dump(classifier, open("classifier.model", "wb"))
np.save("all_data_scaled.npy", all_data_scaled)
np.save("all_labels.npy", all_labels)


# ------------------ SAVE METADATA ----------------------------------------------
# Save mean spectra and max distance to the mean along with the metadata
cls_ids = np.unique(all_labels)[1:]  # Ignore the background
cls_means = [all_data_scaled[all_labels == idx].mean(0) for idx in cls_ids]
cls_dist = [
    np.std(spectral_angle_nd_to_vec(all_data_scaled[all_labels == idx], cls_means[i]))
    for i, idx in enumerate(cls_ids)
]
cls_res = {idx: [cls_means[i], cls_dist[i]] for i, idx in enumerate(cls_ids)}
metadata = {k: [f"Class {k}", cls_res[k][0], cls_res[k][1]] for k in cls_ids}
pickle.dump(metadata, open("metadata.txt", "wb"))
