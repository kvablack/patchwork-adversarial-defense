from patch_database import PatchDatabase
from PIL import Image
import os
from tqdm import tqdm

def main():
    db = PatchDatabase()
    db.print_info()
    images = next(os.walk("data"))[2][1000:1100]
    for f in tqdm(images):
        img = db.create_patchwork(f"data/{f}", print_progress=True)
        Image.fromarray(img).save(f"outpca2/{f}")

if __name__ == "__main__":
    main()
