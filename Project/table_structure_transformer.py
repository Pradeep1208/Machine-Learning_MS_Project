import cv2
import matplotlib.pyplot as plt

image = cv2.imread("dataset/images/page_12.png")  # replace XX

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = cv2.threshold(
    gray,
    150,
    255,
    cv2.THRESH_BINARY_INV
)[1]

# Horizontal lines
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (40, 1)
)

horizontal = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)

# Vertical lines
vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, 40)
)

vertical = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel
)

# Combine
table_mask = cv2.add(horizontal, vertical)

plt.figure(figsize=(10, 8))
plt.imshow(table_mask, cmap="gray")
plt.title("Detected Table Structure")
plt.axis("off")

plt.savefig("output/table_structure.png")
print("Saved: output/table_structure.png")