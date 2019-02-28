import os
import numpy as np
import cv2
import h5py
from annoy import AnnoyIndex
from collections import abc
from tqdm import tqdm

def count_patches(width, height, size, stride):
    """
    Counts the number of patches that will be produced by an image of dimensions `width` by `height`
    with `size` (width, height) and `stride` (horizontal, vertical).
    """
    return ((width - size[0]) // stride[0] + 1) * ((height - size[1]) // stride[1] + 1)

NUM_FEATURES = 3
def generate_features(image, size, stride):
    """
    The function used to extract features from patches for the purpose of patch-matching.
    This is the function we will fine-tune in the future.

    Current features: average hue, average saturation, average value
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    feature_batch = []
    for x0, x1, y0, y1 in convolution_indices(hsv.shape[0], hsv.shape[1], size, stride):
        patch = image[x0:x1, y0:y1]
        feature_batch.append(np.mean(patch, axis=(0, 1)))  # average hue, saturation, and value for each patch
    return feature_batch

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
    Computes the four bouding indices of the nth convolution window for an image.
    nth_convolution_indices(n, width, height, size, stride) is equivalent to list(convolution_indices(width, height, size, stride))[n]
    except it is much faster.
    """
    cols = (width - size[0]) // stride[0] + 1
    row = n // cols
    col = n % cols
    x0 = col * stride[0]
    y0 = row * stride[1]
    return x0, x0 + size[0], y0, y0 + size[1]


class PatchDatabaseGenerator:
    """
    Has the ability to generate a patch database from a directory full of images. The
    patch database will be saved to disk in 2 separate files: one hdf5 file with all of the
    raw image data, and one index file with all of the feature data.
    """


    def __init__(self, img_folder, num=None, size=10, stride=5, n_trees=10):
        """
        Initializes a patch database generator with an image folder, a patch size, and a patch stride.

        Args:
            img_folder: The path to the folder containing the images.
            num: The number of images from this folder to use. If None, all images in the folder
                will be used. (default: None)
            size: The size of the patches to generate, in pixels. Either an int, in which case the
                patches will be square, or a 2-tuple of the form (width, height). (default: 10)
            stride: The stride with which to generate patches, in pixels. Either an int, in which
                case the stride will be the same horizontally and vertically, or a 2-tuple of the form
                (horizontal, vertical). (default: 5)
            n_trees: The number of trees to use for building the index. See Annoy documentation (default: 10)
        """
        self.img_folder = img_folder
        self.n_trees = n_trees

        if isinstance(size, abc.Iterable) and len(size) == 2 and all(isinstance(x, int) for x in stride):
            self.size = tuple(size)
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise ValueError(f"size argument {size} must be int or sequence of ints of length 2")

        if isinstance(stride, abc.Iterable) and len(stride) == 2 and all(isinstance(x, int) for x in stride):
            self.stride = tuple(stride)
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            raise ValueError(f"stride argument {stride} must be int or sequence of ints of length 2")

        try:
            self.file_list = next(os.walk(self.img_folder))[2][:num]
        except StopIteration:
            raise ValueError(f"cannot find image folder at path {img_folder}")

    def _count_pixels(self):
        """Returns (total_pixels, total_patches) with current settings."""
        total_patches = 0
        total_pixels = 0
        for filename in tqdm(self.file_list, "Counting pixels and patches"):
                img = cv2.imread(os.path.join(self.img_folder, filename))
                total_patches += count_patches(img.shape[0], img.shape[1], self.size, self.stride)
                total_pixels += img.shape[0] * img.shape[1] * img.shape[2]
        return total_pixels, total_patches

    def print_info(self):
        """
        Prints out various statistics for the dataset that will be created by `generate()`. Useful
        for figuring out if the current size, stride, and number of images are reasonable.
        """
        print(f"Image folder: {self.img_folder}")
        print(f"Number of images to use: {len(self.file_list)}")
        print(f"Patch size: {self.size[0]}x{self.size[1]}, stride: {self.stride[0]}x{self.stride[1]}")
        print(f"Number of features per patch: {NUM_FEATURES}")
        print(f"Number of trees (for index): {self.n_trees}")
        total_pixels, total_patches = self._count_pixels()
        print("Statistics:")
        print(f"\tTotal number of pixels: {total_pixels:,}")
        print(f"\tTotal number of patches: {total_patches:,}")
        print(f"\tSize of image file on disk: {total_pixels * 3 / 10**6:,.2f} MB")
        print(f"\tSize of features on disk: {total_patches * NUM_FEATURES * 4 / 10**6:,.2f} MB", end="\t")
        print("<-- does not include extra indexing data, which depends on the number of trees and may be very significant")

    def generate(self, images_file="images.hdf5", index_file="index.ann"):
        """
        Generate the patch database, consisting of one hdf5 file with all of the image data, and one
        index file with all of the feature data and an indexing structure for nearest-neighbor
        searching.

        Args:
            images_file: The filename for the image data file. (default: 'images.hdf5')
            index_file: The filename for the index file. (default: 'index.nms')
        """
        total_pixels, _ = self._count_pixels()
        with h5py.File(images_file, 'w') as f:
            image_data = f.create_dataset("image_data", (total_pixels,), dtype=np.uint8)  # actual image data, flattened
            image_pointers = f.create_dataset("image_pointers", (len(self.file_list),), dtype=np.int)  # pointers to mark start of each image
            image_shapes = f.create_dataset("image_shapes", (len(self.file_list), 3), dtype=np.int)  # shape info for each image
            patch_counts = f.create_dataset("patch_counts", (len(self.file_list),), dtype=np.int)  # patch counts for each image
            index = AnnoyIndex(NUM_FEATURES, metric="euclidean")  # index, which will also hold feature data
            index.on_disk_build(index_file)

            patch_counter = 0
            img_data_ptr = 0
            for i, filename in enumerate(tqdm(self.file_list, "Saving images/generating features")):
                img = cv2.imread(os.path.join(self.img_folder, filename))  # read image from disk
                image_shapes[i] = img.shape
                image_pointers[i] = img_data_ptr

                img_flat = img.flatten()
                image_data[img_data_ptr:img_data_ptr + len(img_flat)] = img_flat  # save flattened image array to disk

                feature_batch = generate_features(img, self.size, self.stride)  # generate features
                patch_counts[i] = len(feature_batch)   # save number of patches in this image
                for i, v in enumerate(feature_batch):
                    index.add_item(patch_counter + i, np.array(v, dtype=np.float32))

                patch_counter += len(feature_batch)
                img_data_ptr += len(img_flat)

            f.attrs["size"] = self.size
            f.attrs["stride"] = self.stride
            f.attrs["num_features"] = NUM_FEATURES

            print(f"Building index with {self.n_trees} trees...")
            index.build(self.n_trees)


class PatchDatabase:
    """
    Represents a patch database that has been loaded from disk. Has the ability to turn an image
    into patchwork using this database.

    Attributes:
        size: The patch size, in the form (height, width).
        stride: The stride used to originally generate the patches, in the form (horizontal, vertical).
    """
    def __init__(self, images_file="images.hdf5", index_file="index.ann"):
        """
        Initializes a patch database from an images file and an index file.

        Args:
            images_file: Filename for hdf5 file containing image data (default: 'images.hdf5')
            index_file: Filename for file containing index and feature data (default: 'index.nms')
        """
        self.file = h5py.File(images_file, 'r')

        if self.file.attrs["num_features"] != NUM_FEATURES:
            raise ValueError(f"Image data file {images_file} was created with {self.file.attrs['num_features']} features, whereas feature extraction is currently using {NUM_FEATURES} features. Versions are incompatible.")

        self.size = tuple(self.file.attrs["size"])
        self.stride = tuple(self.file.attrs["stride"])
        self.images_file = images_file
        self.index_file = index_file
        # load patch counts and compute cumulative sums
        self.cumulative_patch_counts = np.cumsum(np.insert(self.file["patch_counts"], 0, 0))

        self.index = AnnoyIndex(NUM_FEATURES, metric="euclidean")
        self.index.load(index_file)

    def print_info(self):
        """
        Prints out various information about the loaded patch database.
        """
        print(f"Patch database loaded from image data file {self.images_file} and index file {self.index_file}")
        print(f"\tNumber of images: {len(self.file['image_shapes'])}")
        print(f"\tPatch size: {self.size[0]}x{self.size[1]}, stride: {self.stride[0]}x{self.stride[1]}")
        print(f"\tNumber of features per patch: {NUM_FEATURES}")
        print(f"\tNumber of trees in index: {self.index.get_n_trees()}")
        print(f"\tTotal number of pixels: {len(self.file['image_data']):,}")
        print(f"\tTotal number of patches: {self.index.get_n_items():,}")

    def create_patchwork(self, image):
        """
        Turns an input image into patchwork.

        Args:
            image: String or numpy array. If it is a string, it will be treated as a filename, and
                the corresponding image will be loaded from disk. If it is a numpy array, it must
                have shape (width, height, channels).

        Returns:
            Numpy array of the form (width, height, channels) representing the patchwork image.
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        # generate features and query database for nearest neighbors for each patch
        feature_batch = generate_features(image, self.size, self.size)
        query_result = [self.index.get_nns_by_vector(v, 1)[0] for v in feature_batch]

        # perform patch substitutions with nearest neighbors
        patchwork = np.empty_like(image)
        patchwork_conv_indices = convolution_indices(image.shape[0], image.shape[1], self.size, self.size)
        for (x0, x1, y0, y1), patch_index in tqdm(list(zip(patchwork_conv_indices, query_result)), "Generating patchwork"):
            # use binary search on cumulative patch counts to find image that patch_index belongs to
            image_index = np.searchsorted(self.cumulative_patch_counts, patch_index, side="right") - 1

            patch_num = patch_index - self.cumulative_patch_counts[image_index]
            data_start = self.file["image_pointers"][image_index]
            width, height, channels = self.file["image_shapes"][image_index]
            tx0, tx1, ty0, ty1 = nth_convolution_indices(patch_num, width, height, self.size, self.stride)

            # read patch from disk column by column
            for x, tx in zip(range(x0, x1), range(tx0, tx1)):
                start = (height * tx + ty0) * channels
                end = (height * tx + ty1) * channels
                patchwork[x, y0:y1, :] = self.file["image_data"][data_start + start:data_start + end].reshape(-1, channels)

        return patchwork
