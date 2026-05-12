import os
import easyocr
from PIL import Image
import xml.etree.ElementTree as ET






# we need to list the directories before we write a for loop to loop over the images

image = os.listdir('../archive/Indian_Number_Plates/Sample_Images/')

for image_name in image:
    xml_name = image_name.replace('.jpg', '.xml')
    xml_path = '../archive/Annotations/Annotations/' + xml_name
    print(xml_path)
# replacing strings

tree = ET.parse(xml_path)

root = tree.getroot()

xmin = int(float(root.find('.//xmin').text))
ymin = int(float(root.find('.//ymin').text))
xmax = int(float(root.find('.//xmax').text))
ymax = int(float(root.find('.//ymax').text))

print(xmin, ymin, xmax, ymax)
