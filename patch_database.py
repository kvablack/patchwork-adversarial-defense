import numpy as np
import h5py
import pickle
from PIL import Image
import os
from tqdm import tqdm
from annoy import AnnoyIndex

from utils import convolution_indices, nth_convolution_indices, num_patches, gen_patches_from_image

class PatchDatabase:
    """
    Represents a patch database that has been loaded from disk. Has the ability to turn an image
    into patchwork using this database.

    Attributes:
        size: The patch size, in the form (height, width).
        stride: The stride used to originally generate the patches, in the form (horizontal, vertical).
    """
    def __init__(self, images_file="images.hdf5", index_file="index.patchdb"):
        """
        Initializes a patch database from an images file and an index file.

        Args:
            images_file: Filename for hdf5 file containing image data (default: 'images.hdf5')
            index_file: Filename for file containing index (default: 'index.patchdb')
        """
        self.images_file = images_file
        self.index_file = index_file
        self.file = h5py.File(images_file, 'r')

        with open(index_file, "rb") as f:
            with open("index.tmp", "wb") as f2:
                while True:
                    data = f.readline()
                    if data == b"BEGIN PICKLE SEGMENT\n":
                        break
                    f2.write(data)
            metadata = pickle.load(f)
            self.pca = pickle.load(f)

        self.size = metadata["size"]
        self.stride = metadata["stride"]
        self.dims = metadata["dims"]

        # load patch counts and compute cumulative sums
        counts = [num_patches(shape[0], shape[1], self.size, self.stride) for shape in self.file["image_shapes"]]
        self.cumulative_patch_counts = np.cumsum(np.insert(np.array(counts), 0, 0))

        self.index = AnnoyIndex(self.dims, metric="euclidean")
        self.index.load("index.tmp")

    def print_info(self):
        """
        Prints out various information about the loaded patch database.
        """
        print(f"Patch database loaded from image data file {self.images_file} and index file {self.index_file}")
        print(f"\tNumber of images: {len(self.file['image_shapes'])}")
        print(f"\tPatch size: {self.size[0]}x{self.size[1]}, stride: {self.stride[0]}x{self.stride[1]}")
        print(f"\tNumber of dimensions per patch: {self.dims}")
        print(f"\tNumber of trees in index: {self.index.get_n_trees()}")
        print(f"\tTotal number of pixels: {len(self.file['image_data']):,}")
        print(f"\tTotal number of patches: {self.index.get_n_items():,}")

    def create_patchwork(self, image, print_progress=False):
        """
        Turns an input image into patchwork.

        Args:
            image: String or numpy array. If it is a string, it will be treated as a filename, and
                the corresponding image will be loaded from disk. If it is a numpy array, it must
                have shape (width, height, channels).
            print_progress: Whether or not to print progress using tqdm (default: False)

        Returns:
            Numpy array of the form (width, height, channels) representing the patchwork image.
        """
        if isinstance(image, str):
            image = np.array(Image.open(image))

        num = num_patches(image.shape[0], image.shape[1], self.size, self.size)
        if print_progress:
            loop = tqdm(desc=f"Querying {num} patches...", total=num)
        patches = np.array([patch.ravel() for patch in gen_patches_from_image(image, self.size, self.size)])
        if self.pca is None:
            transformed = patches
        else:
            transformed = self.pca.transform(patches)
        query_result = [self.index.get_nns_by_vector(v[:self.dims], 1)[0] for v in transformed]

        # perform patch substitutions with nearest neighbors
        patchwork_conv_indices = convolution_indices(image.shape[0], image.shape[1], self.size, self.size)
        patchwork = np.empty_like(image)
        if print_progress:
            loop.iterable = list(zip(patchwork_conv_indices, query_result))
            loop.set_description("Reading patches from disk")
        else:
            loop = zip(patchwork_conv_indices, query_result)
        for (x0, x1, y0, y1), patch_index in loop:
            # use binary search on cumulative patch counts to find image that patch_index belongs to
            image_index = np.searchsorted(self.cumulative_patch_counts, patch_index, side="right") - 1
            patch_num = patch_index - self.cumulative_patch_counts[image_index]

            ptr = self.file["image_pointers"][image_index]
            width, height, channels = self.file["image_shapes"][image_index]
            tx0, tx1, ty0, ty1 = nth_convolution_indices(patch_num, width, height, self.size, self.stride)

            # image = self.file["image_data"][ptr:ptr + width * height * channels].reshape(width, height, channels)
            # patchwork[x0:x1, y0:y1, :] = image[tx0:tx1, ty0:ty1, :]

            # read patch from disk column by column
            for x, tx in zip(range(x0, x1), range(tx0, tx1)):
                start = (height * tx + ty0) * channels
                end = (height * tx + ty1) * channels
                patchwork[x, y0:y1, :] = self.file["image_data"][ptr + start:ptr + end].reshape(-1, channels)

        loop.close()
        return patchwork

    def __del__(self):
        os.remove("index.tmp")
