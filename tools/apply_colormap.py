import os
import cv2
import numpy as np

directory = "depth_image_stuff/depth_images/raw"
output_directory = "depth_image_stuff/depth_images/colormapped"

for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        min_val, max_val = np.min(image), np.max(image)
        normalized_image = ((image - min_val) /
                            (max_val - min_val) * 255).astype(np.uint8)

        colormapped_image = cv2.applyColorMap(
            normalized_image, cv2.COLORMAP_JET)

        colormapped_image[normalized_image == 0] = [0, 0, 0]

        rotated_image = cv2.rotate(colormapped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, rotated_image)
