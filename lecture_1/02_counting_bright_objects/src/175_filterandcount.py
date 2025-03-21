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
    output_path = "lecture_1/01_getting_started/output/175/"
    # output_path_image = output_path + "flower_output_gray.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv.imread(input_path + "DJI_0222.JPG")
    
    dst = cv.GaussianBlur(img, (5, 5), 0)
    cv.imwrite(output_path + "ex05-1-smoothed.jpg", dst)

    # Convert to HSV
    img_hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)
    segmented_image = cv.inRange(img_hsv, (30, 50, 30), (80, 185, 155))
    cv.imwrite(output_path + "ex05-2-hsv_segmented.jpg", segmented_image)

    # Morphological filtering the image
    kernel = np.ones((20, 20), np.uint8)
    closed_image = cv.morphologyEx(segmented_image, cv.MORPH_CLOSE, kernel)
    cv.imwrite(output_path + "ex05-3-closed.jpg", closed_image)

    # Locate contours.
    contours, hierarchy = cv.findContours(closed_image, cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)

    # Draw a circle above the center of each of the detected contours.
    for contour in contours:
        M = cv.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv.circle(img, (cx, cy), 40, (0, 0, 255), 2)

    print("Number of detected balls: %d" % len(contours))

    cv.imwrite(output_path + "ex05-4-located-objects.jpg", img)