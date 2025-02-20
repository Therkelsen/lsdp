import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path_euclidean = "lecture_1/01_getting_started/output/1611/flower_euclidean_distances_image.png"
    input_path_mahalanobis = "lecture_1/01_getting_started/output/1611/flower_mahalanobis_distances_image.png"    
    output_path = "lecture_1/01_getting_started/output/1613/"
    output_path_euclidean_otsu = output_path + "flower_euclidean_otsu_thresholded.png"
    output_path_mahalanobis_otsu = output_path + "flower_mahalanobis_otsu_thresholded.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the images in grayscale mode
    img_euclidean = cv2.imread(input_path_euclidean, cv2.IMREAD_GRAYSCALE)
    img_mahalanobis = cv2.imread(input_path_mahalanobis, cv2.IMREAD_GRAYSCALE)

    # Use Otsu's method to determine the threshold
    threshold_value, segmented_image = cv2.threshold(
        img_euclidean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print("Euclidean threshold value found by Otsu's method")
    print(threshold_value)
    cv2.imwrite(output_path_euclidean_otsu, segmented_image)

    threshold_value, segmented_image = cv2.threshold(
        img_mahalanobis, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print("Mahalanobis threshold value found by Otsu's method")
    print(threshold_value)
    cv2.imwrite(output_path_mahalanobis_otsu, segmented_image)