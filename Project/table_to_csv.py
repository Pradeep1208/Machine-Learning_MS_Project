from PIL import Image, ImageDraw
import torch

from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection
)

# Load image
image_path = "output/tables/page_15.png_table_0.png"
image = Image.open(image_path).convert("RGB")

print("Loading processor...")
processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)

print("Loading model...")
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)

# Prepare input
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert predictions
target_sizes = torch.tensor([image.size[::-1]])

results = processor.post_process_object_detection(
    outputs,
    threshold=0.7,
    target_sizes=target_sizes
)[0]

# Draw predictions
draw = ImageDraw.Draw(image)

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"]
):
    box = [round(i, 2) for i in box.tolist()]

    print(
        f"Label={label.item()} "
        f"Score={score:.3f} "
        f"Box={box}"
    )

    draw.rectangle(box, outline="red", width=2)

output_path = "output/table_structure_transformer.png"
image.save(output_path)

print(f"Saved: {output_path}")