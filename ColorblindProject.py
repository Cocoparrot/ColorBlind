import pygame
import numpy as np
from PIL import Image
from collections import OrderedDict

try:
    import matplotlib as mpl
    _NO_MPL = False
except ImportError:
    _NO_MPL = True

try:
    import pickle
except ImportError:
    import cPickle as pickle
from pkg_resources import parse_version

Main_window = pygame.display.set_mode((800, 800))
Surface = pygame.Surface

# test image for the colorblind test preferable this is changeable in the program menu
image = pygame.image.load()


def transform_colorspace(img, mat):
    """
        Using this to transform the RGB values of the image to a different color space so it is easier to convert the values
        Arguments --> from img: array of shape (M, N, 3)
                           mat: array of shape (3, 3) this is the conversion from on color space to a different space
        Returns --> array of shape (M, N, 3)
    """
    return np.einsum("ij, ...j", mat, img)


def simulate_deficit(img, color_deficit="d"):
    """
        This function is meant to simulate the deficit so it is easier to correct them later on so they are viewable
        to people with the deficit

    :param img: image file --> the image we are going to convert and correct
    :param color_deficit: {d, t, p} --> different types of color_deficit
                                        d = deuternopia
                                        t = trinatopia
                                        p = protonopia
    :return: sim_rgb: array of shape (M, N, 3) --> the simulated image in RGB format
    """

    # Colorspace transformation matrices, these  are the values we need to simulate the deficit
    cb_matrices = {
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
        "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])
    }
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    # Precomputed inverse
    lms2rgb = np.array([[8.09444479e-02, -1.30504409e-01, 1.16721066e-01],
                        [-1.02485335e-02, 5.40193266e-02, -1.13614708e-01],
                        [-3.65296938e-04, -4.12161469e-03, 6.93511405e-01]])

    img = img.copy()
    img = img.convert('RGB')

    rgb = np.asarray(img, dtype=float)
    # first go from RGB to LMS space
    lms = transform_colorspace(rgb, rgb2lms)
    # Make the image have the deficit chosen by the input
    sim_lms = transform_colorspace(lms, cb_matrices[color_deficit])
    # get the LMS values back to RGB to update the image with the corrected values
    sim_rgb = transform_colorspace(sim_lms, lms2rgb)
    return sim_rgb
