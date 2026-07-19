from pdf2image import convert_from_path
import os

pdf_path = "dataset/pdfs/sample_paper.pdf"
output_dir = "dataset/images"

os.makedirs(output_dir, exist_ok=True)

pages = convert_from_path(pdf_path, dpi=200)

for i, page in enumerate(pages):
    image_path = os.path.join(output_dir, f"page_{i+1}.png")
    page.save(image_path, "PNG")
    print(f"Saved: {image_path}")

print(f"Successfully converted {len(pages)} pages.")