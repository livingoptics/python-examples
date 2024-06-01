# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

import cProfile
import logging
import os
import pickle
from typing import Tuple

import cv2
import numpy as np
from matplotlib import cm
from lo.sdk.api.acquisition.data.formats import _BAYER_CONFIG_RGGB
from lo.sdk.api.acquisition.io.open import open as sdk_open
from lo.sdk.tools.analysis.apps import BaseAnalysis
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    import cupy

    xp = cupy
    logging.warning(
        "Cupy is successfully imported. "
        "GPU will be used during the execution of this program."
    )
except ImportError:
    logging.warning(
        "GPU enabled cupy failed to import, falling back to CPU. "
        "The execution may be slow."
    )
    import numpy

    xp = numpy


def spectral_angle_nd_to_vec(spectral_list, reference_spectrum):
    """Calculates the spectral angle difference in radians between each row
        of the spectral list and the reference spectrum.

    Args:
        spectral_list (np.ndarray): shape (N_spectra, N_channels)
        reference_spectrum (np.ndarray): shape (N_channels)

    Returns:
        list of SAM scores (np.ndarray): in radians shape (N_spectra)
    """
    return xp.arccos(
        xp.clip(
            xp.dot(spectral_list, reference_spectrum)
            / xp.linalg.norm(spectral_list, axis=1)
            / xp.linalg.norm(reference_spectrum),
            0,
            1,
        )
    )


# Define visual parameters
bg_colour = (10, 10, 10)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 3


def init_metadata(path):
    if os.path.exists(path):
        metadata = pickle.load(open(path, "rb"))
    reference = None
    if "reference" in list(metadata.keys()):
        reference = np.array(metadata.pop("reference"))
    return metadata, reference


def normalize(scene_frame):
    scene_frame = np.asarray(scene_frame)
    scene_frame = scene_frame - scene_frame.min()
    scene_frame = (scene_frame / np.percentile(scene_frame.flatten(), 97)) * 255
    scene_frame = np.clip(scene_frame, 0, 255)
    scene_frame = scene_frame.astype(np.uint8)
    return scene_frame


def white_balance(frame, white_balance=[1.0, 0.75, 1.2]):
    frame = frame * np.array(white_balance)
    frame = frame / np.max(frame) * 255
    return frame.astype(np.uint8)


def _debayer(frame, info, is_low_res=False):
    """
    Debayer low resolution
    :param frame:
    :param info:

    """
    out_dict: dict = {idx: [] for idx in range(3)}
    [
        out_dict[ci].append((frame[bi] / bsi)[None, :, :, None].astype(frame[bi].dtype))
        for bi, ci, bsi in zip(
            _BAYER_CONFIG_RGGB.bayer_index,
            _BAYER_CONFIG_RGGB.color_index,
            _BAYER_CONFIG_RGGB.fractions,
        )
    ]
    res = np.concatenate([np.sum(np.asarray(v), 0) for i, v in out_dict.items()], -1)

    if is_low_res:
        info.sampling_coordinates = np.asarray(info.sampling_coordinates // 2)
    else:
        res = np.repeat(np.repeat(res, 2, axis=1), 2, axis=2)
    return np.squeeze(res), info


class DetectionAnalysis(BaseAnalysis):
    """
    This class can be used stand alone but was designed to work with the LO analysis tool.

    This show an example of how a simple KNN classifier, can be used to identify objects and substances with Living Optics spatial spectral information.

    """

    def __init__(self, **kwargs):
        super(DetectionAnalysis, self).__init__(**kwargs)
        self.point_finder = None
        self.knn_classifier = None
        self.scaler = None
        self.metadata = None
        self.reference = None

        self.scaler = StandardScaler()
        self.knn_classifier = KNeighborsClassifier(n_jobs=-1)
        self.knn_loaded = False

    def inference(
        self,
        info,
        scene_frame,
        spectra,
        query_radius=50,
        sa_factor=2.5,
        max_neighbours_factor=0.5,
        contour=False,
    ):
        sc = info.sampling_coordinates.astype(np.int32)

        # Apply classifier over spectra.
        # Subtract 1 from labels to match metadata indices
        if self.reference is not None:
            spectra = spectra / self.reference
        scaled_spectra = self.scaler.transform(spectra)
        labels = self.knn_classifier.predict(scaled_spectra) - 1

        frame_vis = scene_frame
        mask_view = np.zeros_like(scene_frame)
        mask_view = cv2.rectangle(
            mask_view, (10, 10), (self.text_width, self.text_height), bg_colour, -1
        )
        seg_mask = frame_vis.copy()
        for k, v in self.metadata.items():
            # Filter false positives using spectral angle
            if v is None:
                continue
            c = self.colours[k]
            thr = v[2] * sa_factor
            seg_mask = spectral_angle_nd_to_vec(scaled_spectra[labels == k], v[1]) < thr
            locs = sc[labels == k][seg_mask]

            # Filter false positives spatially
            tree = KDTree(locs)
            max_dist = tree.query_radius(locs, query_radius)
            max_neighbours = max([len(dist) for dist in max_dist])

            spatial_mask = np.asarray(
                [
                    len(loc) > (max_neighbours * max_neighbours_factor)
                    for loc in max_dist
                ]
            )
            locs = locs[spatial_mask]

            if contour:
                contours = cv2.convexHull(locs)
                if contours is None:
                    continue
                contours = contours[..., ::-1]
                cv2.drawContours(mask_view, [contours], -1, c, cv2.FILLED)

            else:
                for pts in locs[:, ::-1]:
                    cv2.circle(frame_vis, pts, 3, c, 2)
            # Draw legend
            mask_view = cv2.rectangle(
                mask_view, (20, 20 + 40 * k), (40, 40 + 40 * k), self.colours[k], -1
            )
            mask_view = cv2.putText(
                mask_view,
                v[0],
                (50, 40 + 40 * k),
                font,
                font_scale,
                self.colours[k],
                thickness,
            )

        frame_vis[mask_view != 0] = cv2.addWeighted(
            frame_vis[mask_view != 0], 0.3, mask_view[mask_view != 0], 0.7, 1
        )[:, 0]

        return frame_vis

    def setup_model(self, classifier_path, scaler_path, metadata_path):
        print("init_classifier")
        self.knn_classifier = pickle.load(open(classifier_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))
        self.metadata, self.reference = init_metadata(metadata_path)
        self.text_width = (
            max(
                [
                    cv2.getTextSize(v[0], font, font_scale, thickness)[0][0]
                    for k, v in self.metadata.items()
                ]
            )
            + 60
        )
        self.text_height = 40 * len(self.metadata) + 10
        colours = cm.get_cmap("tab10", len(self.metadata)).colors * 255
        self.colours = [tuple([int(k) for k in item]) for item in colours]
        self.knn_loaded = True

    def __call__(
        self,
        loframe: Tuple,
        contour: bool = False,
        query_radius: int = 40,
        sa_factor: float = 2.5,
        max_neighbours_factor: float = 0.6,
        debayer: bool = True,
        R_balance: float = 1.0,
        B_balance: float = 0.75,
        G_balance: float = 1.2,
        downsample_output: bool = False,
        classifier_path: str = "/datastore/lo/share/samples/spectral-detection/classifier.model",
        scaler_path: str = "/datastore/lo/share/samples/spectral-detection/scaler.model",
        metadata_path: str = "/datastore/lo/share/samples/spectral-detection/metadata.txt",
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Run multiclass KNN classifier prediction model

        Args:
            loframe: LOfmt frame tuple object
            contour: (bool) - Whether to draw contours around objects (True) or circles at detection points (False)
            query_radius: (int) - knn tree query radius
            sa_factor: (float) - A thresholding scale factor for spectral angle outlier filtering
            max_neighbours_factor: (float) - A thresholding scale factor for the maximum number of nearest neighbours
            R_balance: (float) - white balance adjustment
            B_balance: (float) - white balance adjustment
            G_balance: (float) - white balance adjustment
            classifier_path: (str) - path the pickled classifier
            scaler_path: (str) - path the pickled scalar
            metadata_path: (str) - path to pickled metadata

        Returns:
            frame: (Tuple[list, np.ndarray, np.ndarray])
            overlay: np.ndarray

        """

        metadata, preview, spectra = loframe

        if not self.knn_loaded:
            self.setup_model(classifier_path, scaler_path, metadata_path)

        preview = normalize(preview)

        if debayer:
            d_scene_frame, metadata = _debayer(
                preview, metadata, is_low_res=downsample_output
            )
            d_scene_frame = white_balance(
                d_scene_frame, [R_balance, G_balance, B_balance]
            )
        else:
            d_scene_frame = np.stack(
                [np.copy(preview), np.copy(preview), np.copy(preview)], axis=-1
            ).squeeze()

        # subsample for speed
        subsampled_spectra = spectra[:, ::4]
        output = self.inference(
            metadata,
            d_scene_frame,
            subsampled_spectra,
            contour=contour,
            query_radius=query_radius,
            sa_factor=sa_factor,
            max_neighbours_factor=max_neighbours_factor,
        )

        return loframe, output


if __name__ == "__main__":
    # Define paths
    scaler_path = "/datastore/lo/share/samples/spectral-detection/scaler.model"
    classifier_path = "/datastore/lo/share/samples/spectral-detection/classifier.model"
    metadata_path = "/datastore/lo/share/samples/spectral-detection/metadata.txt"

    # file path is an .lo file.
    file_path = "/datastore/lo/share/samples/spectral-detection/liquid-segmentation.lo"

    import pstats
    from pstats import SortKey

    routine = DetectionAnalysis()

    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)

    # itterate over .lo file
    with sdk_open(file_path) as f:
        with cProfile.Profile() as pr:
            for i, frame in enumerate(f):
                # run the routine on this frame
                frame, output = routine(
                    frame,
                    scaler_path=scaler_path,
                    classifier_path=classifier_path,
                    metadata_path=metadata_path,
                    downsample_output=True,
                )

                # Uncomment if the scene was upside down
                # output = np.flipud(output)

                cv2.imshow(
                    "Inference", output.astype(np.float32)[:, :, [2, 1, 0]] / 255
                )
                cv2.waitKey(2)

            stats = pstats.Stats(pr).sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(10)
            pr.dump_stats("stats.txt")
