from patch_generator import PatchDatabaseGenerator, PatchDatabase
import cv2

def main():
    generator = PatchDatabaseGenerator("data", num=1000)
    print(generator.size_estimate() / 10**9)
    generator.generate()
    db = PatchDatabase()
    img = db.create_patchwork("obama.jpg")
    cv2.imwrite("obama_patchwork.jpg", img)

if __name__ == "__main__":
    main()
