from PIL import Image, ImageDraw
import torch
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
)

image_path = "output/tables/page_15.png_table_0.png"

print("Loading image...")
image = Image.open(image_path).convert("RGB")

print("Loading processor...")
processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-detection"
)

print("Loading model...")
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])

results = processor.post_process_object_detection(
    outputs,
    threshold=0.7,
    target_sizes=target_sizes
)[0]

draw = ImageDraw.Draw(image)

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"]
):
    box = [round(i, 2) for i in box.tolist()]

    print(
        f"Table detected "
        f"Score={score:.3f} "
        f"Box={box}"
    )

    draw.rectangle(box, outline="red", width=3)

output_path = "output/table_transformer_result.png"

image.save(output_path)

print(f"Saved: {output_path}")