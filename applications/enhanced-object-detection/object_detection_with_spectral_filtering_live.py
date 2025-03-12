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
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8, _debayer
from lo.sdk.api.analysis.classifier import spectral_roi_classifier
from lo.sdk.api.analysis.helpers import (check_new_spectrum,
                                         draw_rois_and_labels,
                                         get_spectra_from_roi, select_roi)
from lo.sdk.api.analysis.metrics import spectral_angles
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.integrations.yolo.helpers import (get_from_huggingface_model,
                                              select_location_cv2_callback)

# ------------------------------ DATA FILEPATHS ------------------------------
# Factory Calibration location
factory_calibration_folder = "/datastore/lo/share/calibrations/latest_calibration"

# Field calibration file
# Needs to be taken manually using GUI and the path provided here
field_calibration_file = None

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
decoder = SpectralDecoder.from_calibration(factory_calibration_folder, field_calibration_file)

params = {"click_location": None}

# list of classes target spectra
target_mean_spectra = None

with LOCamera() as cam:

    # Example sensor settings - set according to your environment
    cam.frame_rate = 10000000
    cam.gain = 100
    cam.exposure = 633333

    while True:
        try:
            frame = cam.get_frame()
            metadata, scene, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)

            shp = np.shape(scene)
            if shp[-1] != 3:
                if np.max(scene) > 2**12:
                    scene = scene >> 4
                if np.max(scene) > 2**8:
                    fraction = 2**12 // 2**8
                    scene = scene / fraction
                debayered = _debayer(np.squeeze(scene), info=None)
                scene_frame = np.asarray(debayered)
            else:
                scene_frame = scene
            scene_frame = cv2.normalize(
                scene_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            scene_frame = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2RGB)

            low_res_frame = scene_frame[::scale_factor, ::scale_factor]
            # fix any memory ordering issues with the low res downsample
            low_res_frame = np.ascontiguousarray(low_res_frame)

            # Apply apple classification on low resolution scene frame for speed
            boxes = model(low_res_frame)
            boxes = boxes[0].boxes.data.detach().cpu().numpy()

            # convert from (left, top, right, bottom) to (left, right, width, height)
            bbox_coordinates = [
                [b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.astype(np.int32)
            ]

            selected_roi = select_roi(
                params["click_location"], np.array(bbox_coordinates), scale_factor
            )
            params["click_location"] = None

            if selected_roi is not None:
                target_mean_spectrum = np.mean(
                    get_spectra_from_roi(
                        spectra, metadata.sampling_coordinates, selected_roi
                    ),
                    axis=0,
                )

                if target_mean_spectra is None:
                    target_mean_spectra = [target_mean_spectrum]

                if multi_class:
                    if check_new_spectrum(
                        target_mean_spectrum, storage_threshold, target_mean_spectra
                    ):
                        target_mean_spectra.append(target_mean_spectrum)
                else:
                    target_mean_spectra = [target_mean_spectrum]

            if target_mean_spectra:
                class_labels, confidences, classified_spectra, segmentations = (
                    spectral_roi_classifier(
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
