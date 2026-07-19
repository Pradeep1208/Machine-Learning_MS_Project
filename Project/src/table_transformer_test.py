from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

print("Loading processor...")
processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-detection"
)

print("Loading model...")
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)

image = Image.open("output/tables/page_15.png_table_0.png")

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

print("Detection complete!")
print(outputs.logits.shape)