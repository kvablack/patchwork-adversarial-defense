from patch_generator import PatchDatabaseGenerator, PatchDatabase
import cv2

def main():
    generator = PatchDatabaseGenerator("data", num=1000)
    generator.print_info()
    # generator.generate()
    # db = PatchDatabase()
    # db.print_info()
    # img = db.create_patchwork("obama.jpg")
    # cv2.imwrite("obama_patchwork.jpg", img)

if __name__ == "__main__":
    main()
