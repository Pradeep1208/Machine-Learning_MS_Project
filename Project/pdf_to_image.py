import cv2
import pytesseract

image_path = "output/tables/page_15.png_table_0.png"

img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)

print("\n===== OCR OUTPUT =====\n")
print(text)