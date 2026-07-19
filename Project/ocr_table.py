import cv2
import os

INPUT_FOLDER = "dataset/images"
OUTPUT_FOLDER = "output/tables"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for image_file in os.listdir(INPUT_FOLDER):

    if not image_file.endswith(".png"):
        continue

    image_path = os.path.join(INPUT_FOLDER, image_file)

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

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

    table_num = 0

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w > 300 and h > 100:

            cropped = image[y:y+h, x:x+w]

            save_path = os.path.join(
                OUTPUT_FOLDER,
                f"{image_file}_table_{table_num}.png"
            )

            cv2.imwrite(save_path, cropped)

            print("Saved:", save_path)

            table_num += 1y