import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load page image
image = cv2.imread("dataset/images/page_20.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary threshold
_, binary = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY_INV
)

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
table_structure = cv2.add(horizontal, vertical)

# Find contours
contours, _ = cv2.findContours(
    table_structure,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    if w > 30 and h > 20:

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Cells")
plt.axis("off")
plt.show()