import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

class Densifier:
    """Class to handle conversion of LO spectral points ordered in (y,x) to a dense
    hypercube.
    """

    def __init__(
        self,
        interp_start_w: float,
        interp_end_w: float,
        interp_start_h: float,
        interp_end_h: float,
        sampling_coordinates: np.ndarray,
    ):
        """Initialises fixed square range for interpolation.

        Args:
            interp_start_w (float): Starting coordinate (width) for interpolation.
            interp_end_w (float): Ending coordinate (width) for interpolation.
            interp_start_h (float): Starting coordinate (height) for interpolation.
            interp_end_h (float): Ending coordinate (height) for interpolation.
            sampling_coordinates (np.ndarray): Fixed coordinates where spectra is
                sampled.
        """
        self.interp_start_w = interp_start_w
        self.interp_end_w = interp_end_w
        self.interp_start_h = interp_start_h
        self.interp_end_h = interp_end_h
        # Change coordinate ordering from (y,x) to (x,y) to agree with mesh grid
        # ordering.
        self.coords = sampling_coordinates[:, ::-1]

        self.X_eval, self.Y_eval = np.meshgrid(
            np.arange(self.interp_start_w, self.interp_end_w),
            np.arange(self.interp_start_h, self.interp_end_h),
        )

    def __call__(self, z_sample):
        """Converts sampled points to a cube using the
            NearestNDInterpolator. Sparse cube is sampled at
            points specified by self.coords and will be
            interpolated using the meshgrid given in the init.

        Args:
            z_sample (np.ndarray): array shape (N, C)
            where N = len(sampling_coordinates) and C is the number of channels

        Returns:
            result: cube interpolated
        """
        interp = NearestNDInterpolator(self.coords, z_sample)
        res = interp(self.X_eval, self.Y_eval)
        return np.asarray(res)
    
class MatplotlibVisualiser:
    def __init__(self):
        return

    def show_image(self, frame):
        """Shows single image using matplotlib.pyplot

        Args:
            frame (np.ndarray): List containing all the RGB images of dimensions HxWx3

        """
        if not isinstance(frame, np.ndarray):
            frame = frame.get()

        plt.imshow(frame)
        plt.show()

    def _calculate_grid_shape(self, n):
        """Return closest factors

        Args:
            n (int): A number

        Returns:
            x, y: Two closest factors of the number
        """
        # finding the square root
        sr = np.ceil(np.sqrt(n)).astype(np.int32)
        return (sr, sr)

    def show_image_grid(self, frames, labels=None):
        """Shows all the images in the list along with their number and label

        Args:
            frames (np.ndarray): List containing all the RGB images of dimentions HxWx3.
            labels (np.ndarray, optional): Text to be shown for each frame.
            It shall have the same size with frames or shall be set to None.
        """
        frames = [
            frame.get() if not isinstance(frame, np.ndarray) else frame
            for frame in frames
        ]

        figure = plt.figure(figsize=(25, 25))
        cols, rows = self._calculate_grid_shape(len(frames))
        for i in range(0, len(frames)):
            figure.add_subplot(rows, cols, i + 1)
            if labels is not None:
                plt.title(f"{labels[i]}")
            plt.axis("off")
            plt.imshow(frames[i])
        plt.show()

    def plot_signal(self, signal, label=None):
        """Plots 1D signal

        Args:
            signal (np.ndarray): Signal to be plotted
            label (str, optional): Label to be used, optional
        """

        plt.plot(signal, label=label)
        plt.legend()
        plt.show()

    def plot_signals(self, signals, labels):
        for signal, label in zip(signals, labels):
            plt.plot(signal, label=label)
        plt.legend()
        plt.show()

def shift_sampling_coordinates(sampling_coordinates):
    """Updates the sampling coordinates returned by LO Calibration with
    respect to the scene covered by the mask and downsamples/upsamples
    them to the given scale

    Args:
        sampling_coordinates (np.ndarray): Array containing sampling coordinates
        scale (int, optional): Scaling factor to downsample/upsample
        coordinates. Defaults to 1

    Returns:
        coords (np.ndarray): Updated coordinates in the (y, x) format.
        (Shape - [N, 2])
        offsets (np.ndarray): Crop-offsets in the form [y1, x1, y2, x2],
        where (x1, y1) is top-left and (x2, y2) is top-right
    """
    y1, y2, x1, x2 = (
        int(sampling_coordinates[:, 0].min()),
        int(sampling_coordinates[:, 0].max()),
        int(sampling_coordinates[:, 1].min()),
        int(sampling_coordinates[:, 1].max()),
    )
    sampling_coordinates[:, 0] = sampling_coordinates[:, 0] - y1
    sampling_coordinates[:, 1] = sampling_coordinates[:, 1] - x1
    offsets = np.asarray([y1, y2, x1, x2]).astype(np.int32)

    return sampling_coordinates, offsets

def rescale_sampling_coordinates(sampling_coordinates, scale=1):
    coords = (sampling_coordinates * scale).astype(np.int32)
    return coords

def min_max_normalize(image, norm_channels_separate=False):
    """Performs a min-max normalization on an image array, either greyscale
    or with channel information.

    Args:
        image (np.ndarray): An array, shape (H, W) or (H, W, C) where
            H is image height, W is image width, C is number of channels.
        norm_channels_separate (Optional, bool, defaults to False): Only applicable to
            a multi-channeled image. If true then will normalise each channel with
            respect to its own statistics. If false then will take the statistics of
            the entire image.

    Returns:
        image (np.ndarray): The normalised image array with same shape as the input.
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

def dynamic_range_normalize(image):
    image = image - image.min()
    image = (image / np.percentile(image.flatten(), 97)) * 255
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)

def get_nearest_wavelength(wavelengths, bv):
    """Returns wavelength number and index nearest to the given value
    from LO's Wavelengths

    Args:
        wavelengths (np.ndarray): List of LO's Wavelengths
        bv (int): Band Value to find the closest wavelength

    Returns:
        value (np.ndarray): Closest wavelength
        idx (int): Index of the wavelength in LO's Wavelengths
    """
    ar = np.asarray(wavelengths)
    idx = (np.abs(ar - bv)).argmin()
    return ar[idx], idx

def get_rgb_wavelength_indices(wavelengths):
    """Returns BGR wavelength indices from LO's Wavelengths

    Args:
        wavelengths (np.ndarray): List of LO's Wavelengths

    Returns:
        BGR Band (np.ndarray): List of BGR bands indices (in that order)
        in LO's Wavelength-Centers
    """
    _, r_ind = get_nearest_wavelength(wavelengths, 625)
    _, g_ind = get_nearest_wavelength(wavelengths, 550)
    _, b_ind = get_nearest_wavelength(wavelengths, 475)

    return np.asarray([r_ind, g_ind, b_ind])
