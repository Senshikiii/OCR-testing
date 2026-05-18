from PIL import Image
import torch
import csv
import time
import editdistance
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from fast_plate_ocr import LicensePlateRecognizer

CSV_PATH = "./ground_truth.csv"
IMAGE_DIR = "../archive/cropped_plates"

device = "cuda" if torch.cuda.is_available() else "cpu"

def clean(text):
    return re.sub(r'[^a-z0-9]', '', text.strip().lower())

def cer(predicted, ground_truth):
    return editdistance.eval(predicted, ground_truth) / max(len(ground_truth), 1)

def load_rows():
    with open(CSV_PATH, newline='') as f:
        return list(csv.DictReader(f))

def benchmark_trocr(model_name, rows):
    print(f"\nLoading {model_name}...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, device_map=None)
    model.to(device)
    model.eval()

    total_cer, exact_matches, total_time = 0, 0, 0

    for row in rows:
        path = f"{IMAGE_DIR}/{row['filename']}"
        truth = clean(row['plate_text'])
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        elapsed = time.time() - start

        pred = clean(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
        c = cer(pred, truth)
        exact = pred == truth
        total_cer += c
        total_time += elapsed
        if exact:
            exact_matches += 1
        print(f"  {row['filename']}: truth={truth} | pred={pred} | CER={c:.3f} | {'✓' if exact else '✗'} | {elapsed:.2f}s")

    n = len(rows)
    return {"model": model_name, "exact": exact_matches, "total": n, "avg_cer": total_cer/n, "avg_time": total_time/n}

def benchmark_fastplate(model_name, rows):
    print(f"\nLoading fast-plate-ocr ({model_name})...")
    recognizer = LicensePlateRecognizer(model_name)

    total_cer, exact_matches, total_time = 0, 0, 0

    for row in rows:
        path = f"{IMAGE_DIR}/{row['filename']}"
        truth = clean(row['plate_text'])

        start = time.time()
        result = recognizer.run(path)
        elapsed = time.time() - start

        # result is a list of PlatePrediction objects
        if result and len(result) > 0:
            pred = clean(result[0].text if hasattr(result[0], 'text') else str(result[0]))
        else:
            pred = ""

        c = cer(pred, truth)
        exact = pred == truth
        total_cer += c
        total_time += elapsed
        if exact:
            exact_matches += 1
        print(f"  {row['filename']}: truth={truth} | pred={pred} | CER={c:.3f} | {'✓' if exact else '✗'} | {elapsed:.2f}s")

    n = len(rows)
    return {"model": f"fast-plate-ocr/{model_name}", "exact": exact_matches, "total": n, "avg_cer": total_cer/n, "avg_time": total_time/n}

rows = load_rows()
results = []
results.append(benchmark_trocr("microsoft/trocr-base-printed", rows))
results.append(benchmark_fastplate("cct-s-v2-global-model", rows))

print("\n" + "="*65)
print(f"{'Model':<35} {'Exact':>8} {'Avg CER':>10} {'Avg Time':>10}")
print("-"*65)
for r in results:
    print(f"{r['model']:<35} {str(r['exact'])+'/'+str(r['total']):>8} {r['avg_cer']:>10.3f} {r['avg_time']:>9.3f}s")
print("="*65)
