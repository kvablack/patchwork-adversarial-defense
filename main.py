from patch_database import PatchDatabase
import cv2
import os
from tqdm import tqdm

def main():
    db = PatchDatabase()
    db.print_info()
    images = next(os.walk("data"))[2][1000:1100]
    for f in tqdm(images):
        img = db.create_patchwork(f"data/{f}", print_progress=True)
        cv2.imwrite(f"outpca2/{f}", img)

if __name__ == "__main__":
    main()
