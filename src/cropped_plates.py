import os
from PIL import Image
import xml.etree.ElementTree as ET

IMG_DIR = "../archive/Indian_Number_Plates/Sample_Images/"
XML_DIR = "../archive/Annotations/Annotations/"
OUT_DIR = "../archive/cropped_plates/"

os.makedirs(OUT_DIR, exist_ok=True)

image_files = [
    filename
    for filename in os.listdir(IMG_DIR)
    if filename.lower().endswith(".jpg")
]

for image_name in image_files:
    img_path = os.path.join(IMG_DIR, image_name)
    xml_name = image_name.replace(".jpg", ".xml")
    xml_path = os.path.join(XML_DIR, xml_name)

    if not os.path.exists(xml_path):
        print("No XML for", image_name)
        continue

    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img = Image.open(img_path).convert("RGB")

    plate_idx = 0
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name != "number_plate":
            continue

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        w, h = img.size
        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h))

        if xmax <= xmin or ymax <= ymin:
            print("Bad box in", xml_name, xmin, ymin, xmax, ymax)
            continue

        crop = img.crop((xmin, ymin, xmax, ymax))

        base, _ = os.path.splitext(image_name)
        out_name = f"{base}_plate{plate_idx}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)
        crop.save(out_path)
        plate_idx += 1

    if plate_idx == 0:
        print("No number_plate object in", xml_name)

