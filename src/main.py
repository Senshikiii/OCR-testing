# running easyocr

import os
from PIL import Image
import easyocr

plate_dir = "../archive/cropped_plates/"

reader = easyocr.Reader(["en"])

plate_files = [
    filename
    for filename in os.listdir(plate_dir)
    if filename.lower().endswith("jpg")

]

print("Total plate images:", len(plate_files))

for plate_name in sorted(plate_files):
    plate_path = os.path.join(plate_dir, plate_name)
    result = reader.readtext(plate_path)

    print("=======", plate_name, "=======")
    if not result:
        print("[no text detected]")
        continue

    for _, text, _ in result:
        print(text)
