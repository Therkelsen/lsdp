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


def main():
    filename = "../input/DJI_0222.JPG"
    img = cv2.imread(filename)

    dst = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("../output/ex05-1-smoothed.jpg", dst)

    # Convert to HSV
    img_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    segmented_image = cv2.inRange(img_hsv, (30, 50, 30), (80, 185, 155))
    cv2.imwrite("../output/ex05-2-hsv_segmented.jpg", segmented_image)

    # Morphological filtering the image
    kernel = np.ones((20, 20), np.uint8)
    closed_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("../output/ex05-3-closed.jpg", closed_image)

    # Locate contours.
    contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

    # Draw a circle above the center of each of the detected contours.
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img, (cx, cy), 40, (0, 0, 255), 2)

    print("Number of detected balls: %d" % len(contours))

    cv2.imwrite("../output/ex05-4-located-objects.jpg", img)



main()
