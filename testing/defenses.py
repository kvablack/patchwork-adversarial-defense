import matplotlib.pyplot as plt
import cv2
import numpy as np


def dummy_perturb(image_in):
    print("perturb")
    return image_in


def dummy_unperturb(image_in):
    print("unperturb")
    return image_in


def gaussian_noise(image_in):
    noise = np.random.normal(0, 0.005, np.shape(image_in))
    noise.reshape(np.shape(image_in))
    dst = image_in + (noise*255).astype(np.uint8)
    # plt.title("gaussian noise")
    # plt.subplot(121), plt.imshow(image_in)
    # plt.subplot(122), plt.imshow(dst)
    # plt.show()
    return dst


def gaussian_denoise(image_in):
    dst = cv2.fastNlMeansDenoisingColored(image_in, None, 10, 10, 7, 21)
    # plt.title("gaussian denoise")
    # plt.subplot(121), plt.imshow(image_in)
    # plt.subplot(122), plt.imshow(dst)
    # plt.show()
    return dst


def blur(image_in):
    pass


def deblur_gan(image_in):
    pass


def bicubic_interpolation(image_in):
    pass


def srgan(image_in):
    pass


def create_patchwork(image_in):
    pass


def patchwork_gan(image_in):
    pass
