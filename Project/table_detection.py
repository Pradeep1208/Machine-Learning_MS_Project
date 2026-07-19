import cv2

image = cv2.imread("dataset/images/page_12.png")  # replace XX

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = cv2.threshold(
    gray,
    150,
    255,
    cv2.THRESH_BINARY_INV
)[1]

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (40, 1)
)

vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, 40)
)

horizontal = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)

vertical = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel
)

table_mask = cv2.add(horizontal, vertical)

contours, _ = cv2.findContours(
    table_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    if w > 200 and h > 100:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

cv2.imwrite("output/table_boundary.png", image)

print("Saved: output/table_boundary.png")