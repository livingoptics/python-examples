import pickle
import cv2
import numpy as np
import time
from matplotlib import cm
from sklearn.neighbors import KDTree

from .reader import LOReader
from shrimp_apps.tools.spectral import spectral_angle_nd_to_vec

# Define paths
scaler_path = "/datastore/lo/share/samples/spectral-detection/scaler.model"
classifier_path = "/datastore/lo/share/samples/spectral-detection/classifier.model"
metadata_path = "/datastore/lo/share/samples/spectral-detection/metadata.txt"
calibration_path = "/datastore/lo/share/samples/spectral-detection/demo-calibration"
file_path = "/datastore/lo/share/samples/spectral-detection/liquid-segmentation.lo"


# Load trained models and metadata
scaler = pickle.load(open(scaler_path, 'rb'))
classifier = pickle.load(open(classifier_path, 'rb'))
metadata = pickle.load(open(metadata_path, 'rb'))

# Initialise LO decoder
reader = LOReader(calibration_path, file_path)

# Define visual parameters
colours = (cm.get_cmap('tab10', len(metadata)).colors * 255)
colours = [tuple([int(k) for k in item]) for item in colours]
bg_colour = (10, 10, 10)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 3
text_width = max([cv2.getTextSize(v[0], font, font_scale, thickness)[0][0]
                  for k, v in metadata.items()]) + 60
text_height = 40 * len(metadata) + 10

reference = None
if 'reference' in list(metadata.keys()):
    reference = np.array(metadata.pop('reference'))

# Iterate over frames
for i in range(len(reader)):
    # Read frame and decompose it
    frame = reader.get_next_frame(is_dynamic_range_normalize=True, is_low_res=True)
    info, scene_view, spectra = frame
    sc = info.sampling_coordinates.astype(np.int32)

    # Apply classifier over spectra.
    # Subtract 1 from labels to match metadata indices
    if reference is not None:
        spectra = spectra / reference
    stime = time.time()
    scaled_spectra = scaler.transform(spectra)
    labels = classifier.predict(scaled_spectra) - 1
    print(time.time()-stime)
    # Filter & Visualise predictions
    mask_view = np.zeros_like(scene_view)

    for k, v in metadata.items():
        # Filter false positives using spectral angle
        thr = v[2] * 2.5
        seg_mask = spectral_angle_nd_to_vec(scaled_spectra[labels == k], v[1]) < thr
        locs = sc[labels == k][seg_mask]

        # Filter false positives spatially
        tree = KDTree(locs)
        max_dist = tree.query_radius(locs, 50)
        max_neighbours = max([len(dist) for dist in max_dist])

        spatial_mask = np.asarray(
            [len(loc) > (max_neighbours * 0.5) for loc in max_dist])
        locs = locs[spatial_mask]

        for pts in locs[:, ::-1]:
            cv2.circle(mask_view, pts, 15, colours[k], -1)

    # Draw legend
    mask_view = cv2.rectangle(mask_view, (10, 10),
                              (text_width, text_height), bg_colour, -1)
    for k, v in metadata.items():
        mask_view = cv2.rectangle(mask_view, (20, 20 + 40 * k),
                                  (40, 40 + 40 * k), colours[k], -1)
        mask_view = cv2.putText(mask_view, v[0], (50, 40 + 40 * k),
                                font, font_scale, colours[k], thickness)

    scene_view[mask_view != 0] = cv2.addWeighted(
        scene_view[mask_view != 0], 0.3, mask_view[mask_view != 0], 0.7, 1)[:, 0]
    # Show
    cv2.imshow("Inference", scene_view[:, :, [2, 1, 0]])
    cv2.waitKey(2)


