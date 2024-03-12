# Importing important libraries
import argparse
import logging

import cv2
import numpy as np
from pysptools.noise import MNF
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Importing the LO's SDK libraries
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.io.open import open as loopen
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8

import utils


def main(args):
    calpath = args.calibration
    calfilter = args.calibrationfilter
    filepath = args.filepath
    frame_idx = args.frame_idx
    atype = args.atype

    decoder = SpectralDecoder.from_calibration(calpath, calfilter)
    sampling_coords = decoder.sampling_coordinates
    coords, offsets = utils.shift_sampling_coordinates(sampling_coords)
    coords = utils.rescale_sampling_coordinates(coords, scale=1 / 3)
    y1, y2, x1, x2 = offsets
    h, w = int((y2 - y1) / 3), int((x2 - x1) / 3)
    bands = decoder.calibration.wavelengths
    r, g, b = utils.get_rgb_wavelength_indices(bands)

    vis = utils.MatplotlibVisualiser()
    dense = utils.Densifier(0, w, 0, h, coords)

    with loopen(filepath) as frames:
        total_frames = len(frames)
        frames.seek(frame_idx)
        frame = frames.read()
        _, scene_frame, spectra = decoder(frame, scene_decoder=LORAWtoRGB8)

    logging.info(f"Total number of frames in the file: {total_frames}")
    logging.info(f"Retrieving frame index {frame_idx} from the file...")
    rgb_preview = scene_frame[y1:y2, x1:x2][::3, ::3]
    scene = rgb_preview.astype(np.uint8)

    dense_cube = dense(spectra)
    bgr_cube = utils.dynamic_range_normalize(dense_cube[:, :, [b, g, r]])

    # Take the White Reference from user
    wr = cv2.selectROI("Select the White Reference", bgr_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    white = np.mean(
        dense_cube[int(wr[1]) : int(wr[1] + wr[3]), int(wr[0]) : int(wr[0] + wr[2])],
        axis=(0, 1),
    )

    # Take the Dark Reference from user
    bg = cv2.selectROI("Select the Background", bgr_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dark = np.mean(
        dense_cube[int(bg[1]) : int(bg[1] + bg[3]), int(bg[0]) : int(bg[0] + bg[2])],
        axis=(0, 1),
    )

    # Calculate Reflectance
    reflectance = (dense_cube - dark) / (white + 10e-6)

    if atype == 1:
        logging.info("Perfroming PCA on the reflectance of the scene...")
        X = np.reshape(reflectance, (h * w, bands.shape[0]))
        pca = PCA()
        cube_pca = pca.fit_transform(X)
        pcaCube = np.reshape(cube_pca, (h, w, bands.shape[0]))
        pca_labels = ["PCA: " + str(i) for i in range(1, 10)]
        vis.show_image_grid(
            [utils.min_max_normalize(pcaCube[:, :, i]) for i in range(0, 9)],
            labels=pca_labels,
        )
    elif atype == 2:
        logging.info("Perfroming MNF on the reflectance of the scene...")
        mnf = MNF()
        mnfCube = mnf.apply(reflectance)
        mnf_labels = ["MNF: " + str(i) for i in range(1, 10)]
        vis.show_image_grid([mnfCube[:, :, i] for i in range(0, 9)], labels=mnf_labels)
    elif atype == 3:
        logging.info("Perfroming LDA on the reflectance of the scene...")
        mk = cv2.selectROI("Select the area for labels", bgr_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask = np.zeros((h, w), dtype=np.int16)
        mask[int(mk[1]) : int(mk[1] + mk[3]), int(mk[0]) : int(mk[0] + mk[2])] = 255
        X = np.reshape(reflectance, (h * w, bands.shape[0]))
        lda = LDA()
        cube_lda = lda.fit_transform(X, mask.reshape(-1))
        ldaCube = np.reshape(cube_lda, (h, w))
        ldaCube = (utils.min_max_normalize(ldaCube) * 255).astype(np.uint8)

        _, new_mask = cv2.threshold(
            ldaCube, int(ldaCube.max() - ldaCube.max() / 3), 255, cv2.THRESH_BINARY
        )
        scene[new_mask == 255] = [255, 0, 0]
        vis.show_image(scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dimensionality Reduction of LO's Hyperspectral Data",
        description="Contains results from three different "
        "methods of analysis - PCA, MNF and LDA.",
    )
    parser.add_argument(
        "-c",
        "--calibration",
        type=str,
        required=True,
        help="Path of the Calibration directory for given Living Optics Camera System.",
    )
    parser.add_argument(
        "-cf",
        "--calibrationfilter",
        type=str,
        required=True,
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
        "--atype",
        type=int,
        default=1,
        required=True,
        help="The type of the analysis you want to implement. "
        "Values must be any from [1, 2, 3], where, "
        "1 = Principle Compenents Analysis (PCA) "
        "2 = Minimum Noise Fraction (MNF), 3 = Linear Discriminant Analysis (LDA).",
    )
    parser.add_argument(
        "-i",
        "--frame_idx",
        type=int,
        default=0,
        required=True,
        help="Frame index in the data file to be evaluated.",
    )
    # Inputs
    args = parser.parse_args()
    # Execute Main
    main(args)