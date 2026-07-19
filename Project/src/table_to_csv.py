import cv2
import pytesseract
import pandas as pd

image_path = "output/tables/page_15.png_table_0.png"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)

lines = []

for line in text.split("\n"):
    line = line.strip()
    if line:
        lines.append([line])

df = pd.DataFrame(lines, columns=["Extracted_Text"])

output_csv = "output/extracted_table.csv"

df.to_csv(output_csv, index=False)

print(f"Saved: {output_csv}")