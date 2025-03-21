import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)

if __name__ == "__main__":
    input_path = "lecture_1/02_counting_bright_objects/input/"
    output_path = "lecture_1/01_getting_started/output/174/"
    # output_path_image = output_path + "flower_output_gray.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv.imread(input_path + "under_exposed_DJI_0213.JPG")
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    segmented_image = cv.inRange(img_lab, (10, 90, 120), (70, 140, 160))
    cv.imwrite(output_path + "ex04_underexposed.jpg", segmented_image)
    compare_original_and_segmented_image(img, segmented_image, "underexposed")

    img = cv.imread(input_path + "well_exposed_DJI_0214.JPG")
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    segmented_image = cv.inRange(img_lab, (100, 90, 120), (200, 140, 200))
    cv.imwrite(output_path + "ex04_wellexposed.jpg", segmented_image)
    compare_original_and_segmented_image(img, segmented_image, "well exposed")

    img = cv.imread(input_path + "over_exposed_DJI_0215.JPG")
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    segmented_image = cv.inRange(img_lab, (100, 90, 120), (255, 140, 200))
    cv.imwrite(output_path + "ex04_overexposed.jpg", segmented_image)
    compare_original_and_segmented_image(img, segmented_image, "over exposed")
    plt.show()
