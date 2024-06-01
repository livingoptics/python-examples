# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

# ----------------------------- IMPORT PACKAGES ------------------------------
import cv2
import numpy as np

# Hack to get around PyQt incompatibility
img = (np.random.random((256, 256)) * 255).astype(np.uint8)
cv2.imshow("dummy", img)
cv2.waitKey(1)
cv2.destroyAllWindows()

from pathlib import Path

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.analysis.classifier import spectral_roi_classifier
from lo.sdk.api.analysis.helpers import (
    check_new_spectrum,
    draw_rois_and_labels,
    get_spectra_from_roi,
    select_roi,
)
from lo.sdk.api.analysis.metrics import spectral_angles
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.integrations.yolo.helpers import (
    get_from_huggingface_model,
    select_location_cv2_callback,
)

# ------------------------------ DATA FILEPATHS ------------------------------
# Calibration location
calibration_folder = "/datastore/lo/share/calibrations/latest_calibration"

# Field calibration frame
calibration_frame_path = None

# ------------------------------ USER PARAMETERS ------------------------------
# Set to True to run in multiclass mode
multi_class = True

# A smaller threshold will result in fewer False negatives, but fewer True positives as well
classification_threshold = 0.095

# Running at a lower resolution by increasing the scale factor will increase the number of frames per second the
# Huggingface model can process.
scale_factor = 2

# Minimum spectral angle to consider a classified spectrum as different from previously classified spectra, and,
# therefore, to assign a new object name
storage_threshold = 0.03

# Proportion of the ROI which must be classified to consider it a true detection
binary_threshold = 1 / 15

# Set true to print the confidences and whether a sufficient number of points within each ROI were detected for a
# valid detection to occur. This is useful for figuring out the correct classification_threshold, storage threshold
# and binary threshold
debug = False

# --------------------------- SETUP YOLO MODEL -------------------------------
model = get_from_huggingface_model("Ultralytics/YOLOv8", "yolov8s.pt")

# -------------------------- CREATE DECODER --------------------=--------------
decoder = SpectralDecoder.from_calibration(calibration_folder, calibration_frame_path)

params = {"click_location": None}

# list of classes target spectra
target_mean_spectra = None

with LOCamera() as cam:
    
    # set sensor settings
    cam.frame_rate = 10000000
    cam.gain = 100
    cam.exposure = 633333

    while True:
        try:
            frame = cam.get_frame()
            metadata, scene, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)

            # flip scene view if upside down
            # scene_frame = np.flipud(scene_frame)

            scene = cv2.normalize(scene, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            low_res_frame = scene[::scale_factor, ::scale_factor]
            # fix any memory ordering issues with the low res downsample
            low_res_frame = np.ascontiguousarray(low_res_frame)

            # Apply face detection on low resolution scene frame for speed
            boxes = model(low_res_frame)
            boxes = boxes[0].boxes.data.detach().cpu().numpy()

            # convert from (left, top, right, bottom) to (left, right, width, height)
            bbox_coordinates = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.astype(np.int32)]


            selected_roi = select_roi(params["click_location"], np.array(bbox_coordinates), scale_factor)
            params["click_location"] = None

            if selected_roi is not None:
                target_mean_spectrum = np.mean(
                    get_spectra_from_roi(spectra, metadata.sampling_coordinates, selected_roi), axis=0
                )

                if target_mean_spectra is None:
                    target_mean_spectra = [target_mean_spectrum]

                if multi_class:
                    if check_new_spectrum(target_mean_spectrum, storage_threshold, target_mean_spectra):
                        target_mean_spectra.append(target_mean_spectrum)
                else:
                    target_mean_spectra = [target_mean_spectrum]

            if target_mean_spectra:
                class_labels, confidences, classified_spectra, segmentations = spectral_roi_classifier(
                    spectra,
                    np.array(target_mean_spectra),
                    spectral_angles,
                    bbox_coordinates,
                    decoder.sampling_coordinates,
                    classification_threshold=classification_threshold,
                    scale_factor=scale_factor,
                    binary_threshold=binary_threshold,
                    debug=debug,
                )
            else:
                class_labels = [0] * len(bbox_coordinates)
                confidences = [b[-1] for b in boxes.astype(np.int32)]

            display = draw_rois_and_labels(
                low_res_frame,
                bbox_coordinates,
                class_labels,
                confidences,
                font_scale=0.5,
                line_thickness=1,
            )

            cv2.imshow("LO Frames", display)
            cv2.setMouseCallback("LO Frames", select_location_cv2_callback, params)
            cv2.waitKey(200)
        except KeyboardInterrupt:
            break
