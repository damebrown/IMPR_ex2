import numpy as np
from math import e as e
from math import pi as pi
from scipy.signal import convolve2d
from imageio import imread
from skimage import img_as_float64
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# ================
# Global variables
# ================
# representation code for a gray scale image
GRAY_REP = 1
# constant to normalize a [0,255] image
NORMALIZE_CONST = 255
# complex i
COMPLEX_I = np.complex(0, 1)
# 2 * pi * i, for the fourier actions
TWO_PI_I = 2 * pi * COMPLEX_I
# the kernel for a horizontal image derivative
DERIVATIVE_KERNEL = np.matrix([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
# basic gaussian kernel
GAUSSIAN_KERNEL = np.matrix([1, 1]).astype("float64")


def dft_matrix_factory(n, is_inverse):
    """
    aid function for the 1d dft and idft functions. initializes and returns the suitable dft matrix
    :param n: the length of the signal
    :param is_inverse: true if the function was called from the IDFT function, false otherwise
    :return: the suitable dft matrix
    """
    if is_inverse:
        fixed_power = TWO_PI_I / n
    else:
        fixed_power = TWO_PI_I * ((-1) / n)
    xx, yy = np.meshgrid(np.arange(0, n), np.arange(0, n))
    e_mat = np.full((n, n), e)
    omega_matrix = fixed_power * xx * yy
    return e_mat ** omega_matrix


def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation.
    :param signal: an array of dtype float64 with shape (N,1).
    :return: complex Fourier signal
    """
    return np.dot(dft_matrix_factory(len(signal), False), signal)


def IDFT(fourier_signal):
    """
    transform a 1D Fourier discrete signal to its normal representation.
    :param fourier_signal: an array of dtype complex128 with shape (N,1).
    :return: complex signal
    """
    return np.dot(dft_matrix_factory(len(fourier_signal), True) / len(fourier_signal), fourier_signal)


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: a grayscale image of dtype float64
    :return: the image's 2d fourier transform
    """
    return DFT(DFT(image).T).T


def IDFT2(fourier_image):
    """
    convert a 2D discrete fourier image signal to its normal representation
    :param fourier_image: a 2D array of dtype complex128
    :return: the fourier_image's original image
    """
    return IDFT(IDFT(fourier_image).T).T


def conv_der(im):
    """
    computes the magnitude of image derivatives in each direction separately.
    :param im: a grayscale images of type float64
    :return: the magnitude of the derivative of type float64
    """
    dx, dy = convolve2d(im, DERIVATIVE_KERNEL, mode = 'same'), convolve2d(im, DERIVATIVE_KERNEL.T, mode = 'same')
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def fourier_der(im):
    """
    computes the magnitude of image derivatives using Fourier transform.
    :param im: a float64 grayscale image to derive
    :return: the magnitude of the derivative of type float64
    """
    im_f = DFT2(im)
    xx, yy = np.meshgrid(np.arange((- len(im[0]) / 2), len(im[0]) / 2), np.arange((- len(im) / 2), len(im) / 2))
    xx, yy = xx * (TWO_PI_I / np.size(im)), yy * (TWO_PI_I / np.size(im))
    dx, dy = IDFT2(im_f * xx), IDFT2(im_f * yy)
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def kernel_factory(kernel_size):
    """
    an aid function for the blurring functions. generates a gaussian kernel of the wanted size.
    :param kernel_size: the wanted size
    :return: a (kernel_size, kernel_size) shape gaussian kernel
    """
    if kernel_size == 1 or kernel_size % 2 == 0:
        return np.matrix([1])
    base_vector = GAUSSIAN_KERNEL
    while len(base_vector[0]) < kernel_size:
        base_vector = convolve2d(base_vector, GAUSSIAN_KERNEL)
    g = convolve2d(base_vector, base_vector.T)
    return g / sum(sum(g))


def blur_spatial(im, kernel_size):
    """
    Performs image blurring using 2D convolution between the image f and a gaussian
    kernel g.
    :param im: the input image to be blurred (grayscale float64 image).
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer).
    :return: the output blurry image (grayscale float64 image).
    """
    if kernel_size == 1:
        return im
    return convolve2d(im, kernel_factory(kernel_size), mode = 'same')


def blur_fourier(im, kernel_size):
    """
    performs image blurring with gaussian kernel in Fourier space.
    :param im: the input image to be blurred (grayscale float64 image).
    :param kernel_size: the size of the gaussian kernel in each dimension (an odd integer).
    :return:
    """
    f_kernel = np.pad(kernel_factory(kernel_size), [(int((len(im) - kernel_size) / 2), ),
                                                    (int((len(im[0]) - kernel_size) / 2), )], mode = 'constant')
    if f_kernel.shape != im.shape:
        if f_kernel.shape[0] != im.shape[0]:
            f_kernel = np.pad(f_kernel, [(0, 1), (0, 0)], mode = 'constant')
        if f_kernel.shape[1] != im.shape[1]:
            f_kernel = np.pad(f_kernel, [(0, 0), (0, 1)], mode = 'constant')
    f_kernel = np.fft.ifftshift(f_kernel)
    f_kernel, f_im = DFT2(f_kernel), DFT2(im)
    return np.real(IDFT2(np.multiply(f_kernel, f_im)))


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: The filename of an image on disk (could be grayscale or RGB).
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
            image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with intensities normalized to [0,1]
    """
    rgb_img = imread(filename)
    rgb_img = img_as_float64(rgb_img)
    if representation == GRAY_REP:
        rgb_img = rgb2gray(rgb_img)
    return rgb_img / NORMALIZE_CONST


def show(im):
    return np.fft.fftshift(np.log(1 + np.abs(DFT2(im))))


plt.figure()
plt.imshow(show(blur_fourier(read_image('p2.jpg', 1), 15)), cmap = 'gray')
plt.show()
plt.figure()
plt.imshow(show(blur_fourier(read_image('p3.jpg', 1), 15)), cmap = 'gray')
plt.show()
plt.figure()
plt.imshow(show(blur_fourier(read_image('p4.jpg', 1), 15)), cmap = 'gray')
plt.show()
plt.figure()
plt.imshow(show(blur_fourier(read_image('p5.jpg', 1), 15)), cmap = 'gray')
plt.show()

# plt.figure()
# plt.figure()
# plt.figure()
# plt.figure()
# plt.imshow(show(read_image('p2.jpg', 1)))
# plt.show()
# plt.imshow(show(read_image('p3.jpg', 1)))
# plt.show()
# plt.imshow(show(read_image('p4.jpg', 1)))
# plt.show()
# plt.imshow(show(read_image('p5.jpg', 1)))
# fig, a = plt.subplots(1, 5)
# a[0].imshow(show(read_image('p.jpg', 1)))
# a[1].imshow(show(read_image('p5.jpg', 1)))
# a[2].imshow(show(read_image('p2.jpg', 1)))
# a[3].imshow(show(read_image('p3.jpg', 1)))
# a[4].imshow(show(read_image('p4.jpg', 1)))
# a[5].imshow(show(read_image('p5.jpg', 1)))
# plt.show()
# s_3 = blur_spatial(im, 25)
# f_3 = blur_fourier(im, 25)
# fig3.suptitle('Size - 49:\n Spatial Blur Spectrum, Image Spectrum, Fourier Blur Spectrum')
# a[0].set_title('Spatial Spectrum'), a[1].set_title('Image Spectrum'), a[2].set_title('Fourier blur Spectrum')

# for image in ['bright.jpg', 'gray.jpg', 'color.jpeg', 'im.jpeg', 'grey.jpg', 'frog.png']:
#     im = read_image(image, 1)
#     s_3 = blur_spatial(im, 25)
#     f_3 = blur_fourier(im, 25)
#     fig3, a = plt.subplots(1, 3)
#     fig3.suptitle('Size - 49:\n Spatial Blur Spectrum, Image Spectrum, Fourier Blur Spectrum')
#     a[0].set_title('Spatial Spectrum'), a[1].set_title('Image Spectrum'), a[2].set_title('Fourier blur Spectrum')
#     a[0].imshow(show(s_3), cmap = 'gray'), a[1].imshow(show(im), cmap = 'gray'), a[2].imshow(show(f_3), cmap = 'gray')
#     plt.show()

