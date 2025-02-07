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
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
segmented_image = cv2.inRange(img, (0, 100, 100), (255, 255, 255))
cv2.imwrite("../output/ex03_underexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "underexposed")
plt.show()

img = cv2.imread("../input/well_exposed_DJI_0214.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
segmented_image = cv2.inRange(img, (0, 100, 200), (255, 255, 255))
cv2.imwrite("../output/ex03_wellexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "well exposed")
plt.show()

img = cv2.imread("../input/over_exposed_DJI_0215.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
segmented_image = cv2.inRange(img, (0, 30, 0), (255, 90, 255))
cv2.imwrite("../output/ex03_overexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "over exposed")
plt.show()

