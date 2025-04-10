# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

# ----------------------------- IMPORT PACKAGES ------------------------------
import cv2
import numpy as np
from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as sdkopen
from pysptools.noise import MNF
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ------------------------------ DATA FILEPATHS ------------------------------
# File to view
filepath = "/datastore/lo/share/samples_v2/macbeth/macbeth.lo"

# ----------------------------- USER INPUTS ----------------------------------
# Frame to perform analysis on
frame_idx = 0

# The analysis type to run. Must be one of:
#   "PCA" = Principle Component Analysis (PCA)
#   "MNF" = Minimum Noise Fraction (MNF)
#   "LDA" = Linear Discriminant Analysis (LDA)
analysis_type = "PCA"

print("Initialising...")

# --------------------------- CALIBRATION ------------------------------------
#decoder = SpectralDecoder.from_calibration(factory_calibration_folder, field_calibration_file)

# ------------------------HELPER FUNCTIONS----------------------------------
def simple_bgr(spectra, wavelengths):
    """
    Extracts BGR wavelength colours from the decoded spectra.
    """
    rgb_idx = [np.argmin(np.abs(wavelengths - w)) for w in [475, 550, 625]]
    return spectra[:, rgb_idx]


def min_max_normalize(image, per_channel=False):
    """
    Performs min-max normalisation on an image.

    Args:
        image (np.ndarray): Input image array of shape (H, W) or (H, W, C).
        per_channel (bool): If True, normalizes each channel separately.

    Returns:
        np.ndarray: Normalized image array.
    """
    # Normalise over the first two dimensions
    if per_channel or image.ndim == 2:
        return (image - image.min(axis=(0, 1))) / (
            image.max(axis=(0, 1)) - image.min(axis=(0, 1))
        )
    # Normalise over all dimensions
    return (image - image.min()) / (image.max() - image.min())


# ------------------ LOAD THE FRAME AND GET SPECTRA-----------------
file = sdkopen(filepath)
file.seek(frame_idx)
frame = file.read()
metadata, scene, spectra = frame
wavelengths = metadata.wavelengths

# --------------------------- UPSAMPLER ------------------------------------
# Instantiate an upsampler to convert from spectral list to an array in the scene view coordinates
upsampler = NearestUpSample(metadata.sampling_coordinates, scale=1 / 3)

dense_cube = upsampler(spectra)

# Generate BGR visualization
bgr = simple_bgr(spectra, wavelengths)
bgr_cube = upsampler(bgr) / np.max(upsampler(bgr))

# Get a white reference from the user
wr = cv2.selectROI("Select the White Reference", bgr_cube)
cv2.waitKey(0)
cv2.destroyAllWindows()
white_ref = np.mean(
    dense_cube[int(wr[1]) : int(wr[1] + wr[3]), int(wr[0]) : int(wr[0] + wr[2])],
    axis=(0, 1),
)

# Get a dark reference from the user
bg = cv2.selectROI("Select the Background", bgr_cube)
cv2.waitKey(0)
cv2.destroyAllWindows()
dark_ref = np.mean(
    dense_cube[int(bg[1]) : int(bg[1] + bg[3]), int(bg[0]) : int(bg[0] + bg[2])],
    axis=(0, 1),
)

# --------------------- CALCULATE REFLECTANCE -------------------
reflectance = (dense_cube - dark_ref) / (white_ref + 1e-6)
h, w, c = reflectance.shape

# --------------------- PERFORM ANALYSIS -------------------
X = reflectance.reshape(h * w, c)

if analysis_type == "PCA":
    print("Performing PCA...")
    X = reflectance.reshape(-1, wavelengths.shape[0])
    transformed = PCA().fit_transform(X)
    result_cube = transformed.reshape(*reflectance.shape)

elif analysis_type == "MNF":
    print("Performing MNF...")
    result_cube = MNF().apply(reflectance)

elif analysis_type == "LDA":
    print("Performing LDA...")
    mk = cv2.selectROI("Select an area of interest", bgr_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = np.zeros(reflectance.shape[:2], dtype=np.int16)
    mask[int(mk[1]) : int(mk[1] + mk[3]), int(mk[0]) : int(mk[0] + mk[2])] = 255

    X = reflectance.reshape(-1, wavelengths.shape[0])
    transformed = LDA().fit_transform(X, mask.ravel())
    result_cube = transformed.reshape(reflectance.shape[:2])

    mask = (min_max_normalize(result_cube) * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, int(np.max(mask) * 0.67), 255, cv2.THRESH_BINARY)

    bgr_cube[mask == 255] = [255, 0, 0]
    cv2.imshow("LDA Result", bgr_cube)
    cv2.waitKey(0)
else:
    print("Error: analysis_type must be 'PCA', 'MNF', or 'LDA'")
    exit()

for i in range(min(9, result_cube.shape[-1])):
    cv2.imshow(f"{analysis_type} Component {i}", min_max_normalize(result_cube[..., i]))
    cv2.waitKey(1000)
cv2.waitKey(0)
