# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os
import pickle
import numpy as np
from matplotlib import cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from lo.sdk.api.acquisition.io.open import open as sdk_open
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.helpers.import_numpy_or_cupy import xp


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
mask_file = "/datastore/lo/share/samples/spectral-detection/segmentation-mask.npy"

# To extend this example to additional lo data a segmentation labeled mask required, in npy format.
# Mask is expected to be a a binary mask of C,Y,X where C is the number of classes.


# ------------------------------ INIT CLASSIFIER ------------------------------
scaler = StandardScaler()
classifier = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    n_jobs=len(os.sched_getaffinity(0)) - 1
)
colours = (cm.get_cmap('tab10', 10).colors * 255)
colours = [tuple([int(k) for k in item]) for item in colours]

#
debayer = True

# ------------------------------ LOAD MASKS ------------------------------------
masks = np.load(mask_file)

# Get first frame and it's spectra

# ------------------------------ READ LO FORMAT FILE ---------------------------
with sdk_open(file_path) as f:
    # Iterate over frames
    
    info, scene_view, spectra = next(f) # Get the first frame in the file for fitting
    

# ------------------------------ OPTIONAL DEBAYERING ---------------------------
if debayer:
    # The example file is bitshifted already
    scene_view = scene_view * 16
    scene_view = LORAWtoRGB8(scene_view)
else:
    scene_view = np.stack(
        [np.copy(scene_view), np.copy(scene_view), np.copy(scene_view)], axis=-1
    ).squeeze()
    scene_view = scene_view.astype(float) / 16

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
fg_labels = [np.zeros(len(s), dtype=np.int32) + np.unique(spectra_indices[i])[1]
             for i, s in enumerate(fg_spectra)]

# ------------------ MERGE SPECTRA AND LABELS ------------------------------------
all_data = np.concatenate((bg_spectra, *fg_spectra), axis=0)
all_labels = np.concatenate((bg_labels, *fg_labels), axis=0)

# ------------------ FIT CLASSIFIER ----------------------------------------------
all_data_scaled = scaler.fit_transform(all_data)
classifier.fit(all_data_scaled, all_labels)

# ------------------ SAVE CLASSIFIER ---------------------------------------------
pickle.dump(scaler, open("scaler.model", 'wb'))
pickle.dump(classifier, open("classifier.model", 'wb'))
np.save("all_data_scaled.npy", all_data_scaled)
np.save("all_labels.npy", all_labels)


# ------------------ SAVE METADATA ----------------------------------------------
# Save mean spectra and max distance to the mean along with the metadata
cls_ids = np.unique(all_labels)[1:]  # Ignore the background
cls_means = [all_data_scaled[all_labels == idx].mean(0) for idx in cls_ids]
cls_dist = [np.std(
    spectral_angle_nd_to_vec(all_data_scaled[all_labels == idx], cls_means[i]))
    for i, idx in enumerate(cls_ids)]
cls_res = {idx: [cls_means[i], cls_dist[i]] for i, idx in enumerate(cls_ids)}
metadata = {
    k: [f"Class {k}", cls_res[k][0], cls_res[k][1]]
    for k in cls_ids
}
pickle.dump(metadata, open("metadata.txt", "wb"))
