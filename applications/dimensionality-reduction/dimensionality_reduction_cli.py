""  # This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import argparse

import cv2
import numpy as np
from lo.sdk.api.acquisition.data.coordinates import NearestUpSample
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.acquisition.io.open import open as sdkopen
from pysptools.noise import MNF
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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


def main(args):
    print("Initialising...")

    # --------------------------- CALIBRATION ------------------------------------
    decoder = SpectralDecoder.from_calibration(
        args.factory_calibration_folder, args.field_calibration_file
    )

    # --------------------------- UPSAMPLER ------------------------------------
    # Instantiate an upsampler to convert from spectral list to an array in the scene view coordinates
    upsampler = NearestUpSample(decoder.sampling_coordinates, scale=1 / 3)

    # ------------------ LOAD THE FRAME AND GET SPECTRA-----------------
    file = sdkopen(args.filepath)
    file.seek(args.frame_index)
    frame = file.read()
    metadata, scene, spectra = decoder(frame, LORAWtoRGB8)
    wavelengths = metadata.wavelengths
    dense_cube = upsampler(spectra)

    # Generate BGR visualisation
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

    if args.analysis_type == "PCA":
        print("Performing PCA...")
        pca = PCA()
        transformed = pca.fit_transform(X)
        result_cube = transformed.reshape(h, w, c)

    elif args.analysis_type == "MNF":
        print("Performing MNF...")
        mnf = MNF()
        result_cube = mnf.apply(reflectance)

    elif args.analysis_type == "LDA":
        print("Performing LDA...")
        mk = cv2.selectROI("Select an area of interest", bgr_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask = np.zeros((h, w), dtype=np.int16)
        mask[int(mk[1]) : int(mk[1] + mk[3]), int(mk[0]) : int(mk[0] + mk[2])] = 255
        lda = LDA()
        transformed = lda.fit_transform(X, mask.ravel())
        result_cube = transformed.reshape(h, w)

        # Apply thresholding
        normalized = min_max_normalize(result_cube) * 255
        _, mask = cv2.threshold(
            normalized.astype(np.uint8),
            int(np.max(normalized) * 0.67),
            255,
            cv2.THRESH_BINARY,
        )

        # Overlay on BGR image
        bgr_cube[mask == 255] = [255, 0, 0]
        cv2.imshow("LDA Result", bgr_cube)
        cv2.waitKey(0)
        return

    # --------------------- DISPLAY RESULTS -------------------
    for i in range(min(9, result_cube.shape[-1])):
        cv2.imshow(
            f"{args.analysis_type} Component {i}",
            min_max_normalize(result_cube[..., i]),
        )
        cv2.waitKey(1000)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dimensionality Reduction of Living Optics hyperspectral data",
        description="Perform PCA, MNF, or LDA on hyperspectral data.",
    )
    parser.add_argument(
        "-c",
        "--calibration-folder",
        type=str,
        required=True,
        help="Path of the Factory Calibration directory for the Living Optics Camera.",
    )
    parser.add_argument(
        "-cf",
        "--calibration-frame-path",
        type=str,
        help="Path of the field calibration file collected from the Living Optics Camera.",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        required=True,
        help="Path of the data file collected from Living Optics Camera.",
    )
    parser.add_argument(
        "-at",
        "--analysis-type",
        choices=["PCA", "MNF", "LDA"],
        default="PCA",
        help="Analysis type.",
    )
    parser.add_argument(
        "-i",
        "--frame-index",
        type=int,
        default=0,
        help="Frame index in the data file to be evaluated.",
    )
    # Inputs
    args = parser.parse_args()
    # Execute Main
    main(args)
