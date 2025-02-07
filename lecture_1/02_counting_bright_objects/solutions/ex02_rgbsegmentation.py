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
# Range adjusted to detection of the green balls.
segmented_image = cv2.inRange(img, (75, 150, 60), (140, 220, 120))
cv2.imwrite("../output/ex02_underexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "underexposed")
plt.show()


img = cv2.imread("../input/well_exposed_DJI_0214.JPG")
segmented_image = cv2.inRange(img, (130, 130, 100), (255, 255, 255))
cv2.imwrite("../output/ex02_wellexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "well exposed")
plt.show()

img = cv2.imread("../input/over_exposed_DJI_0215.JPG")
segmented_image = cv2.inRange(img, (200, 200, 100), (255, 255, 255))
cv2.imwrite("../output/ex02_overexposed.jpg", segmented_image)
compare_original_and_segmented_image(img, segmented_image, "over exposed")
plt.show()

