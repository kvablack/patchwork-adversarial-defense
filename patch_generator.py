import os
import numpy as np
import cv2
import h5py
from annoy import AnnoyIndex
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import pickle
import click

from utils import count_pixels, count_patches, gen_patches_from_image_data

@click.group()
def cli():
    pass

@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("output", type=click.File("wb"))
@click.option("-n", "--num", default=0, type=click.IntRange(min=0, max=None),
              help="Number of images from folder to use (if 0 or not specified, will use all of them)")
@click.option("--dry-run", is_flag=True,
              help="Instead of actually writing the image data, print some useful info and then exit")
def images(directory, output, num, dry_run):
    """
    Takes images from DIRECTORY and writes them to OUTPUT in the hdf5 format.
    """
    if num == 0:
        num = None

    try:
        file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))][:num]
    except StopIteration:
        raise click.ClickException(f"cannot find image folder at path {directory}")

    total_pixels = count_pixels(directory, file_list, print_progress=True)
    if dry_run:
        click.echo(f"Image folder: {directory}")
        click.echo(f"Number of images to use: {len(file_list)}")
        click.echo(f"Total number of pixels: {total_pixels:,}")
        click.echo(f"Estimated final size of image file: {total_pixels / 10**6:,.2f} MB")
    else:
        f = h5py.File(output)
        f.attrs["num"] = len(file_list)
        image_data = f.create_dataset("image_data", (total_pixels,), dtype=np.uint8)  # actual image data, flattened
        image_pointers = f.create_dataset("image_pointers", (len(file_list),), dtype=np.int)  # pointers to mark start of each image
        image_shapes = f.create_dataset("image_shapes", (len(file_list), 3), dtype=np.int)  # shape info for each image

        img_data_ptr = 0
        for i, filename in enumerate(tqdm(file_list, "Writing image data to disk")):
            img = cv2.imread(os.path.join(directory, filename))  # read image from disk
            image_shapes[i] = img.shape
            image_pointers[i] = img_data_ptr

            img_flat = img.ravel()
            image_data[img_data_ptr:img_data_ptr + len(img_flat)] = img_flat  # save flattened image array to disk

            img_data_ptr += len(img_flat)



@cli.command()
@click.argument("imgfile", type=click.File("rb+"))
@click.argument("outfile", type=click.Path(exists=True))
@click.argument("size", type=click.STRING)
@click.argument("stride", type=click.STRING)
@click.option("-n", "--num", default=0, type=click.IntRange(min=0, max=None),
              help="Number of images from image file to use (if 0 or not specified, will use all of them)")
@click.option("-t", "--num-trees", default=10, type=click.IntRange(min=1, max=None),
              help="Number of trees to use for building the Annoy index (default 10)")
@click.option("-p", "--num-components", default=0, type=click.IntRange(min=0, max=None),
              help="Number of components to keep for PCA dimensionality reduction. If 0 or not specified, PCA will be skipped completely and all pixels from each patch will be used.")
@click.option("-b", "--batch-size", default=10000, type=click.IntRange(min=1, max=None),
              help="Batch size to use for fitting and applying the PCA, only has an effect if num-components is set to a nonzero value (default 10000)")
@click.option("--dry-run", is_flag=True,
              help="Instead of actually generating the index, print some useful info and then exit")
def index(imgfile, outfile, size, stride, num, num_trees, num_components, batch_size, dry_run):
    """
    Generates a patch database from IMGFILE, which should be an hdf5 file with all of the image data.
    SIZE is the patch size and STRIDE is the horizontal/vertical stride, which should both be in the format nxm.
    The index will be written to OUTFILE.
    """
    size = tuple(size.split("x"))
    if len(size) != 2:
        raise click.ClickException(f"Malformatted size argument {'x'.join(size)}")
    try:
        size = (int(size[0]), int(size[1]))
    except ValueError:
        raise click.ClickException(f"Malformatted size argument {'x'.join(size)}")

    stride = tuple(stride.split("x"))
    if len(stride) != 2:
        raise click.ClickException(f"Malformatted stride argument {'x'.join(stride)}")
    try:
        stride = (int(stride[0]), int(stride[1]))
    except ValueError:
        raise click.ClickException(f"Malformatted size argument {'x'.join(size)}")

    f = h5py.File(imgfile)
    num_images = f.attrs["num"]
    if num != 0 and num > num_images:
        click.echo(f"Warning: there are only {num_images:,} images in the image file {imgfile}, so less than {num:,} images are going to be used.")
    if (num == 0 or num > num_images):
        num = num_images

    if num_components == 0:
        skip = True
        num_components = size[0] * size[1] * f['image_shapes'][0][2]
    else:
        skip = False

    total_patches = count_patches(f["image_shapes"][:num], size, stride)
    if dry_run:
        click.echo(f"Image file: {imgfile}")
        click.echo(f"Number of images to use: {num}")
        click.echo(f"Patch size: {size[0]}x{size[1]}, stride: {stride[0]}x{stride[1]}")
        click.echo(f"Number of trees (for index): {num_trees}")
        click.echo(f"Number of components to keep: {'all' if skip else num_components}")
        click.echo("Statistics:")
        click.echo(f"\tTotal number of patches: {total_patches:,}")
        click.echo(f"\tEstimated size of vectors on disk: {total_patches * num_components * 4 / 10**6:,.2f} MB", nl=False)
        click.echo("\t<-- does not include extra indexing data, which depends on the number of trees and may be very significant")
    else:
        patches = gen_patches_from_image_data(f["image_data"], f["image_pointers"][:num], f["image_shapes"][:num], size, stride)
        index = AnnoyIndex(num_components, metric="euclidean")  # index, which will also hold feature data
        index.on_disk_build(outfile)
        if skip:
            pca = None
            for i, patch in tqdm(enumerate(patches), "Loading patches into index", total=total_patches):
                index.add_item(i, patch)
        else:
            click.echo(f"Fitting PCA on {total_patches:,} vectors with {size[0] * size[1] * f['image_shapes'][0][2]:,} dimensions...")
            pca = IncrementalPCA(n_components=num_components)
            for i in tqdm(range(0, total_patches, batch_size), "Overall progress"):
                # batch = [next(patches) for _ in range(batch_size)]
                loop = tqdm(range(min(batch_size, total_patches - 1)), "Loading patches into memory")
                batch = [next(patches) for _ in loop]
                # loop.set_description("Performing PCA fit...") doesn't work
                pca.partial_fit(batch)

            click.echo(f"Performing PCA transform down to {num_components} dimensions...")
            patches = gen_patches_from_image_data(f["image_data"], f["image_pointers"], f["image_shapes"], size, stride)
            for i in tqdm(range(0, total_patches, batch_size), "Overall progress"):
                # batch = [next(patches) for _ in range(batch_size)]
                batch = []
                for _ in tqdm(range(min(batch_size, total_patches - i)), "Loading patches into memory"):
                    batch.append(next(patches))
                out = pca.transform(batch)
                for j, k in enumerate(range(i, min(i + batch_size, total_patches))):
                    index.add_item(k, out[j][:num_components])

        click.echo(f"\nBuilding index on {total_patches:,} vectors of size {num_components} with {num_trees} trees...")
        index.build(num_trees)

        # save the pca info as well as additional metadata
        with open(outfile, "ab") as f:
            f.write(b"\nBEGIN PICKLE SEGMENT\n")
            metadata = {
                "dims": num_components,
                "size": size,
                "stride": stride
            }
            pickle.dump(metadata, f)
            pickle.dump(pca, f)



if __name__ == "__main__":
    cli()
