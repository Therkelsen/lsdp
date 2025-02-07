import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)

img = cv2.imread("../input/under_exposed_DJI_0213.JPG")
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
segmented_image = cv2.inRange(img_lab, (10, 90, 120), (70, 140, 160))
cv2.imwrite("../output/ex04_underexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "underexposed")
plt.show()

img = cv2.imread("../input/well_exposed_DJI_0214.JPG")
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
segmented_image = cv2.inRange(img_lab, (100, 90, 120), (200, 140, 200))
cv2.imwrite("../output/ex04_wellexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "well exposed")
plt.show()

img = cv2.imread("../input/over_exposed_DJI_0215.JPG")
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
segmented_image = cv2.inRange(img_lab, (100, 90, 120), (255, 140, 200))
cv2.imwrite("../output/ex04_overexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "over exposed")
plt.show()

