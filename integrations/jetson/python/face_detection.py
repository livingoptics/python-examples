# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import os
import cv2
import jetson_inference
import jetson_utils

from lo.sdk.api.acquisition.io.open import open as lo_open
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.helpers.path import getdatastorepath


# ### Setup

PATH_TO_LO_RAW_FILE = os.path.join(
    getdatastorepath(),
    "lo",
    "share",
    "samples",
    "face-spoofing",
    "face-spoof-demo.lo-raw"
)

# Define facenet detector
net = jetson_inference.detectNet("facenet", threshold=0.01)

RESIZE_FACTOR = 0.5


# - We instantiate jetson-inference's detectNet with the facenet model

with lo_open(PATH_TO_LO_RAW_FILE) as f:
    for (encoded_info, encoded_frame), (scene_info, scene_frame) in f:
        # Convert loraw -> RGB8 -> opencv's BGR
        scene_frame = LORAWtoRGB8(scene_frame)
        scene_frame = cv2.cvtColor(scene_frame, cv2.COLOR_RGB2BGR)
        
        # The particular video we're using has been recorded upside down and so we'll flip
        scene_frame = cv2.flip(scene_frame, 0)

        # Place scene_frame into cuda for jetson inference
        scene_frame_cuda = jetson_utils.cudaFromNumpy(scene_frame)
        
        # Resize image
        scene_frame_cuda_resized = jetson_utils.cudaAllocMapped(
            width=scene_frame_cuda.width * RESIZE_FACTOR,
            height=scene_frame_cuda.height * RESIZE_FACTOR,
            format=scene_frame_cuda.format,
        )

        jetson_utils.cudaResize(scene_frame_cuda, scene_frame_cuda_resized)
        
        # Run Jetson inference
        detections = net.Detect(scene_frame_cuda_resized, overlay="box,labels,conf")

        # Convert scene_frame back into numpy format
        display = jetson_utils.cudaToNumpy(scene_frame_cuda_resized)

        cv2.imshow("Preview", display)

        # Break on 'Esc' key
        key = cv2.waitKey(20)
        if key == 27:
            break

cv2.destroyAllWindows()


# - Instantiate LO's `open` context with the path to the `.loraw` file.
# - Convert to RGB8 using `LORAWtoRGB8` then to BGR for OpenCV's format.
# - Run the detection using jetson-inference.
# - Render the scene with detection overlays. 
