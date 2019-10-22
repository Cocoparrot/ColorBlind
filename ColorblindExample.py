from __future__ import print_function, division

from collections import OrderedDict

import pygame

try:
    import pickle
except ImportError:
    import cPickle as pickle
from pkg_resources import parse_version

from PIL import Image
import numpy as np

assert parse_version(np.__version__) >= parse_version('1.9.0'), "numpy >= 1.9.0 is required for daltonize"

try:
    import matplotlib as mpl
    _NO_MPL = False
except ImportError:
    _NO_MPL = True


def transform_colorspace(img, mat):
    """
        Using this to transform the RGB values of the image to a different color space so it is easier to convert the values
        Arguments --> from img: array of shape (M, N, 3) or mat: array of shape (3, 3) this is the conversion from on color space to a different space
        Returns --> array of shape (M, N, 3)
    """
    return np.einsum("ij, ...j", mat, img)


def simulate(img, color_deficit="d"):
    """
        Simulate the color blindness on the image so it easier to make the conversion to a corrected image

        Arguments --> img: PIL.PngImagePlugin.pngImageFile, input image we use to correct
                      color_deficit : {"d", "p", "t"}, optional --> these are the arguments to select which type we want
                        d for deuteropia
                        p for protonapia
                        t for trinatopia

        Returns --> sim_rgb : array of shape (M, N, 3)
                        simulated image in RGB format
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


def daltonize(rgb, color_deficit='d'):
    """
        Adjust the colors in the image so the are compensated for the color blindness.
    :param rgb: array of shape (M, N, 3)
                original image in RGB format
    :param color_deficit: {"d", "p", "t"}, optional
                type of colorblindness, d for deuternopia (default),
                                        p for prtonapia,
                                        t for tritanopia
    :return:
        dtpn: array of shape (M, N, 3)
            image in RGB format to display with the adjusted color values
    """

    sim_rgb = simulate(rgb, color_deficit)
    err2mod = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    # rgb - sim_rgb is what people with deuternopia cannot see. err2mod corrects it to something they can see more easy.
    rgb = rgb.convert('RGB')
    err = transform_colorspace(rgb - sim_rgb, err2mod)
    dtpn = err + rgb
    return dtpn


def array_to_img(arr):
    """
        Function to convert the numpy array we make to a PIL image.
    :param arr: array of shape (M, N, 3)

    :return: img : PIL.Image.Image
                RGB image created from the array
    """
    # Make sure the values are in the range [0, 255] so that they are RGB values
    arr = np.clip_array(arr)
    arr = arr.astype('uint8')
    img = Image.fromarray(arr, mode='RGB')
    return img


def clip_array(arr, min_value=0, max_value=255):
    """
        Function for the array_to_img function to clip all the values to RGB values
    :param arr: array with the image values
    :param min_value: float, optional --> default 0 because this is the lowest RGB value
    :param max_value: float, optional --> default 255 because this is the highest RGB value
    :return: arr: array with the clipped values for making the new RGB image
    """
    compare_arr = np.ones_like(arr)
    arr = np.maximum(compare_arr * min_value, arr)
    arr = np.minimum(compare_arr * max_value, arr)
    return arr


def get_child_colors(child, mpl_colors):
    """
        Enter all the colors from a MatPlotLib objects into a dictionary

    :param child: MatPlotLib object that we use for the colors
    :param mpl_colors: Ordered Dictionary from collections to store the colors
    :return: mpl_colors Dictionary we make in the function
    """
    mpl_colors[child] = OrderedDict()
    if hasattr(child, "get_color"):
        mpl_colors[child]['color'] = child.get_color()
    if hasattr(child, "get_facecolor"):
        mpl_colors[child]['fc'] = child.get_facecolor()
    if hasattr(child, "get_edgecolor"):
        mpl_colors[child]['ec'] = child.get_edgecolor()
    if hasattr(child, "get_markerfaceolor"):
        mpl_colors[child]['mfc'] = child.get_markerfacecolor()
    if hasattr(child, "get_markeredgecolor"):
        mpl_colors[child]['mec'] = child.get_markeredgecolor()
    if hasattr(child, "get_markerfaceoloralt"):
        mpl_colors[child]['mfcalt'] = child.get_markerfacecolor()
    if isinstance(child, mpl.image.AxesImage):
        mpl_colors[child]['cmap'] = child.get_cmap()
        img_properties = child.properties()
        try:
            img_arr = img_properties['array']
            if len(img_arr.shape) == 3:
                mpl_colors[child]['array'] = np.array(img_arr)
        except KeyError:
            pass
    if hasattr(child, "get_children"):
        grandchildren = child.get_children()
        for grandchild in grandchildren:
            mpl_colors = get_child_colors(grandchild, mpl_colors)
    return mpl_colors


def get_mpl_colors(fig):
    """

    :param fig: matplotlib.figure.Figure --> figure we are getting the colors from
    :return:
    """
    mpl_colors = OrderedDict()
    children = fig.get_children()
    for child in children:
        mpl_colors = get_child_colors(child, mpl_colors)
    return mpl_colors


def get_key_colors(mpl_colors, rgb, alpha):
    """
    From the ordered dictionary of the colors of all the figure object children we made above
    fill the RGB and alpha channel information

    :param mpl_colors: OrderedDict --> dictionary with the colors of the children, matplotlib instances are keys.
    :param rgb: array of shape (M, 1, 3) --> this stores all the RGB colors we have so far.
    :param alpha: array of shape (M,!) --> Here we store all the alpha values we have so far.

    :return: rgb   : array of shape (M+n, 1, 3)
             alpha : array of shape (M+n, 1)
    """

    if _NO_MPL is True:
        raise ImportError("matplotlib not found, can only deal with pixel images")
    cc = mpl.colors.ColorConverter()
    color_keys = ("color", "fc", "ec", "mfc", "mec", "mfcalt", "cmap", "array")
    for color_key in color_keys:
        try:
            color = mpl_colors[color_key]
            # clause to skip unset colors otherwise they are filled in black
            if isinstance(color, str) and color == 'none':
                continue
            if isinstance(color, mpl.colors.LinearSegmentedColormap):
                rgba = color(np.arrange(color.N))
            elif isinstance(color, np.ndarray) and color_key == "array":
                color = color.reshape(-1, 3) / 255
                a = np.zeros((color.shape[0], 1))
                rgba = np.hstack((color, a))
            else:
                rgba = cc.to_rgba_array(color)
            rgb = np.append(rgb, rgba[:, :3])
            alpha = np.append(alpha, rgba[:, 3])
        except KeyError:
            pass
        for key in mpl_colors.keys():
            if key in color_keys:
                continue
            rgb, alpha = get_key_colors(mpl_colors[key], rgb, alpha)
    return rgb, alpha


def arrays_from_dict(mpl_colors):
    """
    Create rgb and alpha arrays from color dictionary>

    :param mpl_colors: OrderedDict --> dictionary with the colors of all children, matplotlib instances are keys

    :return:
    rgb : array of shape (M, 1, 3) --> RGB values of colors in a line image, M is the total number of non-unique colors
    alpha : array of shape (M, 1) --> alpha channel values of all mpl instances
    """
    rgb = np.array([])
    alpha = np.array([])
    for key in mpl_colors.keys():
        rgb, alpha = get_key_colors(mpl_colors[key], rgb, alpha)
    m = rgb.size // 3
    rgb = rgb.reshape((m, 1, 3))
    return rgb, alpha


def set_colors_from_array(instance, mpl_colors, rgba, i=0):
    """
        Set object instance colors to the modified ones in rgba
    :param instance: 
    :param mpl_colors: 
    :param rgba: 
    :param i: 
    :return: 
    """
    cc = mpl.colors.ColorConverter()
    color_keys = ("color", "fc", "ec", "mfc", "mec", "mfcalt", "cmap", "array")
    for color_key in color_keys:
        try:
            color = mpl_colors[color_key]
            if isinstance(color, mpl_colors.LinearSegmentedColormap):
                j = color.N
            elif isinstance(color, np.ndarray) and color_key == "array":
                j = color.shape[0] * color.shape[1]
            else:
                # clause to skip unset colors, otherwise they will be filled in with black
                if isinstance(color, str) and color == 'none':
                    continue
                color_shape = cc.to_rgba_array(color).shape
                j = color_shape[0]
            target_color = rgba[i: i + j, :]
            if j == 1:
                target_color = target_color[0]
            i += j
            if color_key == "color":
                instance.set_color(target_color)
            elif color_key == "fc":
                instance.set_facecolor(target_color)
            elif color_key == "ec":
                instance.set_edgecolor(target_color)
            elif color_key == "mfc":
                instance.set_markerfacecolor(target_color)
            elif color_key == "mec":
                instance.set_markeredgecolor(target_color)
            elif color_key == "mfcalt":
                instance.set_markerfacealtcolor(target_color)
            elif color_key == "cmap":
                instance.cmap.from_list(instance.cmap.name + "_dlt", target_color)
            elif color_key == "array":
                target_color = (target_color.reshape((color.shape[0], color.shape[1], -1)))
                target_color = (target_color[:, :, :3] * 255).astype('uint8')
                instance.set_data(target_color)
        except KeyError:
            pass
    return i


def set_mpl_colors(mpl_colors, rgba):
    """
        set the colors in a color dictionary to new values in rgba.

    :param mpl_colors:  dictionary with all the colors of all the children, MatPlotLib instances are keys
    :param rgba: array of shape (M, 1, 4) containing rgb, alpha channels
    :return:
    """

    i = 0
    for key in mpl_colors.keys():
        i = set_colors_from_array(key, mpl_colors[key], rgba, i)


def prepare_for_transform(fig):
    """
    Gather the color keys and info for mpl figure and arrange them so that the image daltonize routine can be called
    :param fig:
    :return:
    """

    mpl_colors = get_mpl_colors(fig)
    rgb, alpha = arrays_from_dict(mpl_colors)
    return rgb, alpha, mpl_colors


def join_rgb_alpha(rgb, alpha):
    """
    combine (m, n, 3) rgb and (m, n) alpha array into (m, n, 4) rgba.
    :param rgb:
    :param alpha:
    :return:
    """
    rgb = clip_array(rgb, 0, 1)
    r, g, b = np.split(rgb, 3, 2)
    rgba = np.concatenate((r, g, b, alpha.reshape(alpha.size, 1, 1)), axis=2).reshape(-1, 4)
    return rgba


def daltonize_mpl(fig, color_deficit='d', copy=False):
    """
    Daltonize a matplotlib figure.
    :param fig: matplotlib.figur.Figure --> the figure we are going to convert
    :param color_deficit: types of color deficit for all the options see other comments
    :param copy: bool, optional --> should daltonization happen on a copy (True) or the original(False, default)
    :return: daltonized figure
    """
    if copy:
        pfig = pickle.dumps(fig)
        fig = pickle.loads(pfig)
    rgb, alpha, mpl_colors = prepare_for_transform(fig)
    dtpn = daltonize(array_to_img(rgb * 255), color_deficit) / 255
    rgba = join_rgb_alpha(dtpn, alpha)
    set_mpl_colors(mpl_colors, rgba)
    fig.canvas.draw()
    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("output_image", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--daltonize",
                       help="adjust image color palette for color blindness",
                       action="store_true")
    parser.add_argument("-t", "--type", type=str, choices=["d", "p", "t"],
                        help="type of color blindness (deuteranopia, "
                             "protanopia, tritanopia), default is deuteranopia "
                             "(most common)")
    args = parser.parse_args()

    if args.simulate is False and args.daltonize is False:
        print("No action specified, assume daltonizing")
        args.daltonize = True
    if args.type is None:
        args.type = "d"

    orig_img = Image.open(args.input_image)

    if args.daltonize:
        dalton_rgb = daltonize(orig_img, args.type)
        dalton_img = array_to_img(dalton_rgb)
        dalton_img.save(args.output_image)


"""
    todo pygamemenu
    In this segment we are going to make a menu where you can choose which deficit you want to use the algorithm for.
    If i feel very frisky i might make it so you can also upload a different image.
    https://github.com/ppizarror/pygame-menu    

"""