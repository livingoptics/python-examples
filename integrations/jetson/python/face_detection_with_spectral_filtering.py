# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.


import cv2
import jetson_inference
import jetson_utils
import os
import json
import numpy as np
import imutils

from lo.sdk.api.acquisition.io.open import open as lo_open
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.analysis.classifier import spectral_roi_classifier
from lo.sdk.api.analysis.helpers import draw_rois_and_labels
from lo.sdk.api.analysis.metrics import spectral_angles
from lo.sdk.helpers.path import getdatastorepath


# Setup

PATH_TO_LO_RAW_FILE = os.path.join(
    getdatastorepath(),
    "lo",
    "share",
    "samples",
    "face-spoofing",
    "face-spoof-demo.loraw"
)

factory_calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "samples", "face-spoofing", "demo-calibration-face-spoofing")
SPECTRA_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "real_face_spectra.json")

# Define facenet detector
net = jetson_inference.detectNet("facenet", threshold=0.001)

# Instantiate the spectral decoder
decoder = SpectralDecoder.from_calibration(factory_calibration_folder)

# Load target spectra
with open(SPECTRA_JSON_PATH) as f:
    target_spectra = np.asarray(json.load(f)["crosshair"]["y"])

# Threshold parameters
classification_threshold = 0.06
binary_threshold = 1 / 15

RESIZE_FACTOR = 0.5


# - Define jetson-inference's detectNet
# - Instantiate the spectral decoder object for run spectral analysis
# - Load target spectral data from a JSON file which is of the target face.
# - Set classification threshold parameters

# ### Run detection and spectral filtering

with lo_open(PATH_TO_LO_RAW_FILE) as f:
    for frame in f:
        # Decodes encoded_frame and converts scene_frame from LORAW to RGB8 format
        metadata, scene_frame, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)
        
        # RGB -> OpenCV's BGR
        scene_frame = cv2.cvtColor(scene_frame, cv2.COLOR_RGB2BGR)
        
        # The particular video we're using has been recorded upside down and so we'll flip
        scene_frame = cv2.flip(scene_frame, 0)

        # Place scene_frame into cuda for jetson inference
        scene_frame_cuda = jetson_utils.cudaFromNumpy(scene_frame)
                
        # Run Jetson inference
        detections = net.Detect(scene_frame_cuda, overlay="box,labels,conf")

        # Get bounding boxes' coordinates
        bbox_coordinates = [
            [int(obj.Left), int(obj.Top), int(obj.Right) - int(obj.Left), int(obj.Bottom) - int(obj.Top)]
            for obj in detections
        ]

        bbox_coordinates_vertically_flipped = [
            [int(obj.Left), scene_frame.shape[0] - int(obj.Bottom), int(obj.Right) - int(obj.Left), int(obj.Bottom) - int(obj.Top)]
            for obj in detections
        ]

        # Use spectral information to classify real vs face faces
        class_labels, confidences, classified_spectra, segmentations = spectral_roi_classifier(
            spectral_list=spectra,
            target_spectra=target_spectra,
            metric=spectral_angles,
            rois=bbox_coordinates_vertically_flipped,
            sampling_coordinates=decoder.sampling_coordinates,
            classification_threshold=classification_threshold,
            binary_threshold=binary_threshold,
            scale_factor=1,
        )

        # Add classified bounding boxes overlays to scene
        display = draw_rois_and_labels(
            scene=scene_frame,
            rois=bbox_coordinates,
            class_idxs=class_labels,
            confidence=confidences,
            colours=[(0, 0, 255), (0, 255, 0)],
            class_labels={0: "Fake face", 1: "Real face"}
        )

        # Resize
        display = imutils.resize(display, int(display.shape[1]*RESIZE_FACTOR))
        
        # Render
        cv2.imshow("Preview", display)

        # Break on 'Esc' key
        key = cv2.waitKey(20)
        if key == 27:
            break

cv2.destroyAllWindows()


# - The decoder takes the frames as input and outputs decoded spectra and formatted scene_frame according to the given formatter passed into the decoder.
# - Using the bounding boxes from jetson-inference, use the `spectral_roi_classifier` method to classify the bounding boxes against the target spectra.
# - Pass the output to the `draw_rois_and_labels` method to add classified bounding boxes' information as overlay on top of the scene_frame.
