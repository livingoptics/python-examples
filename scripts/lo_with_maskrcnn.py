# ----------------------------- IMPORT PACKAGES ------------------------------
import cv2
import torch
from lo.sdk.helpers.import_numpy_or_cupy import xp
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as lo_open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.api.acquisition.data.formats import _debayer
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def percentile_norm(im: xp.ndarray, low: int = 1, high: int = 95) -> xp.ndarray:
    """
    Normalise the image based on percentile values.

    Args:
        im (xp.ndarray): The input image.
        low (int): The lower percentile for normalization.
        high (int): The higher percentile for normalization.

    Returns:
        xp.ndarray: The normalised image.
    """
    im[..., 0] = im[..., 0] - xp.percentile(im[::100, ::10, 0], low)
    im[..., 0] = im[..., 0] / xp.percentile(im[::100, ::10, 0], high)
    im[..., 1] = im[..., 1] - xp.percentile(im[::100, ::10, 1], low)
    im[..., 1] = im[..., 1] / xp.percentile(im[::100, ::10, 1], high)
    im[..., 2] = im[..., 2] - xp.percentile(im[::100, ::10, 2], low)
    im[..., 2] = im[..., 2] / xp.percentile(im[::100, ::10, 2], high)
    return xp.clip(im, 0, 1) * 255

# ------------------------------ DATA FILEPATHS ------------------------------
# Calibration location
calibration_folder = "/datastore/lo/share/calibrations/latest_calibration"
# If you are running from workstation set this to None.

# Field calibration frame
calibration_frame_path = None

# File to load - pass None to stream directly from the camera
file = "/datastore/lo/share/data/single-potato-peel-flesh-thin-peel-20240924-142815-386193.lo"
if file:
    lo_file = lo_open(file, "r")
else:
    lo_file = None

# ------------------------------ USER PARAMETERS ------------------------------

# Running at a lower resolution by increasing the scale factor will increase the number of frames per second the
# model can process.
scale_factor = 1

# Set a threshold for a true detection
detection_threshold = 0.45

# -------------------------- CREATE DECODER -----------------------------------
if calibration_folder is not None:
    decoder = SpectralDecoder.from_calibration(calibration_folder, calibration_frame_path)


# ------------------------- Setup Mask RCNN Model ----------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights, progress=False).to(DEVICE)
model = model.eval()

with LOCamera(file=lo_file) as cam:

    # set sensor settings
    cam.frame_rate = 10000000
    cam.gain = 100
    cam.exposure = 633333

    while True:
        try:
            frame = cam.get_frame()
            if len(frame) == 4:
                metadata, scene, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)
            else:
                metadata, scene, spectra = frame
                scene = _debayer(np.squeeze(scene), metadata)

            # Colour correct scene view
            scene = percentile_norm(scene.astype(np.float32), high=99)

            # Down sample scene for faster inference
            low_res_frame = scene[::scale_factor, ::scale_factor]

            # Fix any memory ordering issues with the low res down sample
            low_res_frame = np.ascontiguousarray(low_res_frame)

            # Transpose to [C, H, W] format and divide by 255 so the scene is normalised between 0 and 1
            low_res_frame = np.transpose(low_res_frame, [2, 0, 1]) / 255

            # Convert to torch tensor for inference
            low_res_frame = [torch.Tensor(low_res_frame)]

            # Run Mask RCNN on the low resolution frame
            output = model(low_res_frame)[0]

            idxs = output["scores"] >= detection_threshold
            masks = torch.squeeze(output["masks"][idxs] > 0.5)
            # boxes = output["boxes"][idxs]

            display = draw_segmentation_masks(
                low_res_frame[0],
                masks=masks,
                alpha=0.6,
                colors="blue",
            )

            # Transpose back to [H, W, C] format
            display = np.transpose(display.detach().numpy(), [1, 2, 0])[:, :, ::-1]
            cv2.imshow("LO Frames", display)
            cv2.waitKey(200)
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
