"""
This spectral detection tool that runs sklearn classifiers to classify objects.

This class has been designed to run with the Living Optics `analysis` tool. You can try it with some Living Optics
[sample data](../../data-samples-overview.md) using the command below.

Or simply omit the `--calibration-path` and `--file` arguments to try it with a live Living Optics Camera.

Commandline:
    ```cmd
    python analysis.py
        --file /datastore/lo/share/samples/face-spoofing/face-spoof-demo.lo-raw
        --calibration-path /datastore/lo/share/samples/face-spoofing/demo-calibration-face-spoofing
        --analysis spectral_detection.DetectionAnalysis`
    ```

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../install-guide.md#quick-install)

Tips:
    Data locations

    - Input : /datastore/lo/share/samples/liquids-detection/liquid-detection-demo.loraw
    - Input : /datastore/lo/share/samples/liquid-detection/demo-calibration
    - Output : None

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install scikit-learn'
"""


from typing import Tuple

import cv2
import cProfile
import logging
from matplotlib import cm
import os
import numpy as np
from lo.sdk.api.acquisition.data import Calibration
from lo.sdk.tools.analysis.apps.spectral_decode import SpectralDecode
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.preprocessing import StandardScaler

from lo.sdk.api.acquisition.data.formats import _BAYER_CONFIG_RGGB
from lo.sdk.api.acquisition.io.open import open as sdk_open

import pickle
from sklearn.neighbors import KDTree

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
        metadata = pickle.load(open(path, 'rb'))
    reference = None
    if 'reference' in list(metadata.keys()):
        reference = np.array(metadata.pop('reference'))
    return metadata, reference

def normalize(scene_frame):
    scene_frame = np.asarray(scene_frame)
    scene_frame = scene_frame - scene_frame.min()
    scene_frame = (
    scene_frame / np.percentile(scene_frame.flatten(), 97)
    ) * 255
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
        out_dict[ci].append(
            (frame[bi] / bsi)[None, :, :, None].astype(frame[bi].dtype)
        )
        for bi, ci, bsi in zip(
            _BAYER_CONFIG_RGGB.bayer_index,
            _BAYER_CONFIG_RGGB.color_index,
            _BAYER_CONFIG_RGGB.fractions,
        )
    ]
    res = np.concatenate(
        [np.sum(np.asarray(v), 0) for i, v in out_dict.items()], -1
    )

    if is_low_res:
        info.sampling_coordinates = np.asarray(info.sampling_coordinates // 2)
    else:
        res = np.repeat(np.repeat(res, 2, axis=1), 2, axis=2)
    return np.squeeze(res), info


class DetectionAnalysis(SpectralDecode):
    """
    This class can be used stand alone but was designed to work with the LO analysis tool.

    The purpose of this class is to run a Hugging face model on the scene view of the Living Optics Camera to perform object
    detection. Spectral classification is then run to enhance the object detection accuracy. For example, spectral
    classification is robust to face-spoofing, whereas conventional, deep learning models operating on RGB images, are
    not.

    By default, this class loads a face detection model.
    """

    def __init__(self, **kwargs):
        super(DetectionAnalysis, self).__init__(**kwargs)
        self.upsampler = None
        self.point_finder = None
        self.knn_classifier = None
        self.scaler = None
        self.metadata = None
        self.reference = None

        self.scaler = StandardScaler()
        self.knn_classifier = KNeighborsClassifier(n_jobs=-1)
        self.knn_loaded = False



    def init(self, calibration: Calibration, **kwargs):
        super().init(calibration)
    
    def inference(self, info, scene_frame, spectra, query_radius=50, sa_factor=2.5, max_neighbours_factor=0.5, contour=False):
        
        sc = info.sampling_coordinates.astype(np.int32)

        # Apply classifier over spectra.
        # Subtract 1 from labels to match metadata indices
        if self.reference is not None:
            spectra = spectra / self.reference
        print(spectra.shape)
        scaled_spectra = self.scaler.transform(spectra)
        labels = self.knn_classifier.predict(scaled_spectra) - 1
        
        frame_vis = scene_frame
        mask_view = np.zeros_like(scene_frame)
        mask_view = cv2.rectangle(mask_view, (10, 10),
                        (self.text_width, self.text_height), bg_colour, -1)
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
                [len(loc) > (max_neighbours * max_neighbours_factor) for loc in max_dist])
            locs = locs[spatial_mask]

            if contour:
                contours = cv2.convexHull(locs)
                if contours is None:
                    continue
                contours = contours[..., ::-1]
                #for idx, line in enumerate(contours):
                #    mask_view = cv2.line(
                #        mask_view, line[0], contours[idx - 1][0],
                #        [c[0]-40, c[1]-40, c[2]-40], 4)
                cv2.drawContours(mask_view, [contours], -1, c, cv2.FILLED)
                
            else:
                for pts in locs[:, ::-1]:
                    cv2.circle(frame_vis, pts, 3,
                                c,
                                2)
            # Draw legend
            mask_view = cv2.rectangle(mask_view, (20, 20 + 40 * k),
                                    (40, 40 + 40 * k), self.colours[k], -1)
            mask_view = cv2.putText(mask_view, v[0], (50, 40 + 40 * k),
                                    font, font_scale, self.colours[k], thickness)

        frame_vis[mask_view != 0] = cv2.addWeighted(frame_vis[mask_view != 0], 0.3, mask_view[mask_view != 0], 0.7, 1)[:, 0]

        return frame_vis


    def __call__(
        self,
        frame: Tuple,
        spectrum: np.ndarray,
        contour: bool = False,
        query_raduis: int = 40,
        sa_factor: float = 2.5,
        max_neighbours_factor: float = 0.6,
        R_balance: float = 1.0,
        B_balance: float = 0.75,
        G_balance: float = 1.2,
        classifier_path: str = "classifier.model",
        scaler_path: str = "scaler.model",
        state_path: str = "metadata.txt",
        **kwargs,
    ) -> Tuple[list, np.ndarray, np.ndarray]:


        """
        Run multiclass spectral classification on ROIs detected by a Hugging face model.
        Args:
            frame: Tuple
            spectrum: (array_like) - target spectrum to classify against
            measure: (Enum) - selected spectral classification metric
            scale_factor: (int) - amount of down-sampling to apply to the scene view before running the Hugging face
                model on it. Larger number = Faster inference
            classification_threshold: (float) - Threshold above/below (depending on selected metric type) to consider a
                pixel part of the foreground class
            repo_id: (str) - Hugging face model repo ID.
            file_name: (str) - Hugging face model name within the Hugging face repo
            store_classified_spectra: (bool) - whether to store the spectra of classified objects to classify against
                later
            storage_threshold: (float) - minimum spectral angle difference between a newly classified spectrum and all
                previously classified spectra, in order for it to be stored. (Setting this too low will result in
                storing the spectrum of the same object multiple times due to lighting variation between frames - this
                could use a lot of memory).
            **kwargs:

        Returns:
            frame: (Tuple[list, np.ndarray, np.ndarray])
            spectra: (array_like)
            bounding_box_overlay: (array_like)

        """
        if not self.knn_loaded:
            print('init_classifier')
            self.knn_classifier = pickle.load(open(classifier_path, 'rb'))
            self.scaler = pickle.load(open(scaler_path, 'rb'))
            self.metadata, self.reference = init_metadata(state_path)
            self.text_width = max([cv2.getTextSize(v[0], font, font_scale, thickness)[0][0]
                  for k, v in self.metadata.items()]) + 60
            self.text_height = 40 * len(self.metadata) + 10
            colours = (cm.get_cmap('tab10', len(self.metadata)).colors * 255)
            self.colours = [tuple([int(k) for k in item]) for item in colours]
            self.knn_loaded = True



        (encoded_frame_info, encoded_frame), (scene_frame_info, scene_frame) = frame
        frame = [[encoded_frame_info, encoded_frame], [scene_frame_info, scene_frame]]

        metadata, scene_frame, spectra = self.spectral_decoder(frame)

        print(scene_frame.shape)

        #scene_frame = np.flipud(scene_frame)

        scene_frame = normalize(scene_frame)
        d_scene_frame, info = _debayer(scene_frame, metadata, is_low_res=True)
        d_scene_frame = white_balance(d_scene_frame, [R_balance, G_balance, B_balance])
        output = self.inference(metadata, d_scene_frame, spectra, contour=contour, query_radius=query_raduis, sa_factor=sa_factor, max_neighbours_factor=max_neighbours_factor)

        frame[1][1] = output
        return frame, spectra, np.zeros_like(output).mean(-1)

if __name__ == "__main__":
    
    file = '/Users/alex/Documents/round-2-liquid-pouring-1422023039565-20240222-165741-200964.loraw'
    calibration = '/Users/alex/data/latest_calibration'
    import time, pstats
    from pstats import SortKey


    calibration = Calibration(calibration)

    routine = DetectionAnalysis()
    routine.init(calibration)

    with sdk_open(file) as f:
        with cProfile.Profile() as pr:
            for i, frame  in enumerate(f):
                data = routine(frame, None)
                if i == 10:
                    break
            stats = pstats.Stats(pr).sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(10)
            pr.dump_stats('stats.txt')



