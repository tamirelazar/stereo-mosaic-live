from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def create_guassian_filter_vec(filter_size):
    """
    creates a row vector to convolve with in the gaussian pyramid func
    :param filter_size: odd integer
    :return: a numpy array of filter_size size with the normalized binomial coefficients
    """
    filter_vec = np.array([1, 1], dtype=np.float64)
    return_filter = np.array([1, 1], dtype=np.float64)

    while return_filter.shape[0] < filter_size:
        return_filter = np.convolve(return_filter, filter_vec)

    sum = np.sum(return_filter)
    return_filter /= sum

    return return_filter.reshape((1, filter_size))


def reduce(im, filter_vec):
    """
    reduces a N x M image in size, after blurring
    :param im: the image to reduce
    :param filter_vec: the vector to convolve for blurring
    :return: an image of size N/2 x M/2
    """
    filter_size = filter_vec.shape[1]
    filter_mat = np.zeros((filter_size, filter_size))
    filter_mat[filter_size // 2, :] = filter_vec[0, :]
    cov_im = convolve(im.copy(), filter_mat)
    cov_im = convolve(cov_im, filter_mat.T)
    new_im = cov_im[0::2, 0::2]
    return new_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: an odd scalar that represents a squared filter
    :return:
    """
    g_pyr = [im]
    g_filter = create_guassian_filter_vec(filter_size)
    for i in range(max_levels - 1):
        r_im = reduce(im.copy(), g_filter)
        if r_im.shape[0] < 16 or r_im.shape[1] < 16:
            break
        else:
            g_pyr.append(r_im)
            im = r_im

    return g_pyr, g_filter


def read_image(filename, representation):
    """
    read an image file and return as representation
    :param filename: the filename
    :param representation: 1 or 2, representing grayscale/rgb
    :return: a np.float64 matrix representing the image
    """
    image = plt.imread(filename)
    represent = int2float(image)
    if representation == 1 and represent.ndim == 3:
        represent = rgb2gray(represent)
    return represent


def int2float(image):
    """
    takes an image array from (0, 255) to (0, 1)
    :param image: an image array
    :return: an image array of identical shape
    """
    return image.copy().astype(np.float64) / 255
