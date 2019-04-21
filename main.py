from patch_database import PatchDatabase
from PIL import Image
import os
from tqdm import tqdm

def main():
    db = PatchDatabase("images.hdf5", "1000_10x10_5x5_t10.patchdb")
    db.print_info()
    images = sorted(next(os.walk("data/val"))[2])
    for f in tqdm(images):
        img = db.create_patchwork(f"data/val/{f}", k=-1, print_progress=True)
        Image.fromarray(img).save(f"data/val_patches/{f}")

if __name__ == "__main__":
    main()

