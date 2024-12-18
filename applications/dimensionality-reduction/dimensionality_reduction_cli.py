# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

# ----------------------------- IMPORT PACKAGES ------------------------------
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


def main(args):
    calibration_folder = args.calibration_folder
    calibration_frame_path = args.calibration_frame_path
    filepath = args.filepath
    frame_idx = args.frame_index
    analysis_type = args.analysis_type

    # --------------------------- CALIBRATION ------------------------------------
    decode = SpectralDecoder.from_calibration(calibration_folder, calibration_frame_path)

    # --------------------------- UPSAMPLER ------------------------------------
    # Instantiate an upsampler to convert from spectral list to an array in
    # the scene view coordinates
    upsampler = NearestUpSample(decode.sampling_coordinates, scale=1 / 3)


    # ------------------------HELPER FUNCTIONS----------------------------------
    def simple_bgr(spectra, wavelengths):
        """
        Extract the BGR wavelength colors from the decoded spectra
        """
        rgb_idx = [np.argmin(np.abs(wavelengths - w)) for w in [475, 550, 625]]

        return spectra[:, rgb_idx]


    def min_max_normalize(image, norm_channels_separate=False):
        """Performs a min-max normalization on an image array, either greyscale
        or with channel information.

        Args:
            image (xp.ndarray): An array, shape (H, W) or (H, W, C) where
                H is image height, W is image width, C is number of channels.
            norm_channels_separate (Optional, bool, defaults to False): Only applicable to
                a multi-channeled image. If true then will normalise each channel with
                respect to its own statistics. If false then will take the statistics of
                the entire image.

        Returns:
            image (xp.ndarray): The normalised image array with same shape as the input.
        """
        # Normalise over the first two dimensions
        if norm_channels_separate or len(image.shape) == 2:
            image = (image - image.min(axis=(0, 1))) / (
                image.max(axis=(0, 1)) - image.min(axis=(0, 1))
            )
        # Normalise over all dimensions
        else:
            image = (image - image.min()) / (image.max() - image.min())
        return image


    # ------------------ LOAD THE FRAME AND GET SPECTRA-----------------
    file = sdkopen(filepath)

    file.seek(frame_idx)
    frame = file.read()

    metadata, scene, spectra = decode(frame, LORAWtoRGB8)
    wavelengths = metadata.wavelengths

    dense_cube = upsampler(spectra)

    bgr = simple_bgr(spectra, metadata.wavelengths)
    bgr_cube = upsampler(bgr)
    bgr_cube = bgr_cube / bgr_cube.max()


    # Get a white reference from the user
    wr = cv2.selectROI("Select the White Reference", bgr_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    white = np.mean(
        dense_cube[int(wr[1]) : int(wr[1] + wr[3]), int(wr[0]) : int(wr[0] + wr[2])],
        axis=(0, 1),
    )

    # Get a dark reference from the user
    bg = cv2.selectROI("Select the Background", bgr_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dark = np.mean(
        dense_cube[int(bg[1]) : int(bg[1] + bg[3]), int(bg[0]) : int(bg[0] + bg[2])],
        axis=(0, 1),
    )

    # --------------------- CALCULATE REFLECTANCE -------------------
    reflectance = (dense_cube - dark) / (white + 10e-6)
    (h, w, c) = reflectance.shape

    # --------------------- PERFORM ANALYSIS -------------------
    if analysis_type == "PCA":
        print("Perfroming PCA on the reflectance of the scene...")
        X = np.reshape(reflectance, (h * w, wavelengths.shape[0]))
        pca = PCA()
        cube_pca = pca.fit_transform(X)
        pcaCube = np.reshape(cube_pca, (h, w, wavelengths.shape[0]))

        # Display the first components
        for i in range(0, 9):
            cv2.imshow(
                f"PCA: {i}",
                min_max_normalize(cv2.cvtColor(pcaCube[:, :, i], cv2.COLOR_GRAY2BGR)),
            )
            cv2.waitKey(1000)
        cv2.waitKey(0)

    elif analysis_type == "MNF":
        print("Perfroming MNF on the reflectance of the scene...")
        mnf = MNF()
        mnfCube = mnf.apply(reflectance)

        # Display the first components
        for i in range(0, 9):
            cv2.imshow(f"MNF: {i}", min_max_normalize(mnfCube[:, :, i]))
            cv2.waitKey(1000)
        cv2.waitKey(0)

    elif analysis_type == "LDA":
        print("Perfroming LDA on the reflectance of the scene...")
        mk = cv2.selectROI("Select the area of suspected bruise", bgr_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask = np.zeros((h, w), dtype=np.int16)
        mask[int(mk[1]) : int(mk[1] + mk[3]), int(mk[0]) : int(mk[0] + mk[2])] = 255
        X = np.reshape(reflectance, (h * w, wavelengths.shape[0]))
        lda = LDA()
        cube_lda = lda.fit_transform(X, mask.reshape(-1))
        ldaCube = np.reshape(cube_lda, (h, w))
        ldaCube = (min_max_normalize(ldaCube) * 255).astype(np.uint8)

        _, new_mask = cv2.threshold(
            ldaCube, int(ldaCube.max() - ldaCube.max() / 3), 255, cv2.THRESH_BINARY
        )

        # Colour mask on bgr imae and display
        bgr_cube[new_mask == 255] = [255, 0, 0]
        cv2.imshow("LDA", bgr_cube)
        cv2.waitKey(0)
    else:
        print("Error: expected analysis_type to be one of 'PCA', 'MNF' or 'LDA'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dimensionality Reduction of Living Optics hyperspectral data",
        description="Contains implementations for three different "
        "analysis methods - PCA, MNF and LDA.",
    )
    parser.add_argument(
        "-c",
        "--calibration-folder",
        type=str,
        required=True,
        help="Path of the Calibration directory for given Living Optics Camera System.",
    )
    parser.add_argument(
        "-cf",
        "--calibration-frame-path",
        type=str,
        default=None,
        required=False,
        help="Path of the 600nm data file collected from Living Optics Camera System.",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        required=True,
        help="Path of the data file collected from Living Optics Camera System.",
    )
    parser.add_argument(
        "-at",
        "--analysis-type",
        choices=["PCA", "MNF", "LDA"],
        default="PCA",
        help="The type of the analysis you want to implement. "
        "Values must be any from [PCA, MNF, LDA], where, "
        "PCA = Principle Compenents Analysis"
        "MNF = Minimum Noise Fraction, LDA = Linear Discriminant Analysis.",
    )
    parser.add_argument(
        "-i",
        "--frame-index",
        type=int,
        default=0,
        required=False,
        help="Frame index in the data file to be evaluated.",
    )
    # Inputs
    args = parser.parse_args()
    # Execute Main
    main(args)
