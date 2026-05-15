from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model_name = "microsoft/trocr-large-printed"

processor = TrOCRProcessor.from_pretrained(model_name)

model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



def read_text(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image,return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

print(read_text("../archive/cropped_plates/Datacluster_number_plates (101)_plate0.jpg"))


