import numpy as np
import os
import cv2
from tqdm import tqdm

def convolution_indices(width, height, size, stride):
    """
    Generator to produce indices that convolve over an entire image.

    Args:
        width: Image width
        height: Image height
        size: Size of convolution window, 2-tuple of the form (width, height)
        stride: Stride of convolution window, 2-tuple of the form (horizontal, vertical)

    Yields:
        (x0, x1, y0, y1): The four indices that bound the rectangle of the current convolution window.
            x0 and y0 are inclusive, x1 and y1 are exclusive.
    """
    for y in range(0, height - size[1] + 1, stride[1]):
        for x in range(0, width - size[0] + 1, stride[0]):
            yield (x, x + size[0], y, y + size[1])

def nth_convolution_indices(n, width, height, size, stride):
    """
    Computes the four bounding indices of the nth convolution window for an image.
    nth_convolution_indices(n, width, height, size, stride) is equivalent to list(convolution_indices(width, height, size, stride))[n]
    except it is much faster.
    """
    cols = (width - size[0]) // stride[0] + 1
    row = n // cols
    col = n % cols
    x0 = col * stride[0]
    y0 = row * stride[1]
    return x0, x0 + size[0], y0, y0 + size[1]

def gen_patches_from_image(image, size, stride):
    for x0, x1, y0, y1 in convolution_indices(image.shape[0], image.shape[1], size, stride):
        yield image[x0:x1, y0:y1]

def gen_patches_from_image_data(image_data, image_pointers, image_shapes, size, stride):
    for ptr, shape in zip(image_pointers, image_shapes):
        width, height, channels = shape
        image = np.array(image_data[ptr:ptr + width * height * channels]).reshape(width, height, channels)
        for patch in gen_patches_from_image(image, size, stride):
            yield patch.ravel()

def count_pixels(img_folder, file_list, print_progress=False):
    total_pixels = 0
    if print_progress:
        loop = tqdm(file_list, "Counting pixels")
    else:
        loop = file_list
    for filename in loop:
        img = cv2.imread(os.path.join(img_folder, filename))
        total_pixels += img.shape[0] * img.shape[1] * img.shape[2]
    loop.close()
    return total_pixels

def num_patches(width, height, size, stride):
    """
    Counts the number of patches that will be produced by an image of dimensions `width` by `height`
    with `size` (width, height) and `stride` (horizontal, vertical).
    """
    return ((width - size[0]) // stride[0] + 1) * ((height - size[1]) // stride[1] + 1)

def count_patches(img_shapes, size, stride):
    total_patches = 0
    for shape in img_shapes:
        total_patches += num_patches(shape[0], shape[1], size, stride)
    return total_patches
