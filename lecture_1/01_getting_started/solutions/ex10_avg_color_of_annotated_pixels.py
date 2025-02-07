import cv2
import numpy as np

def main():
    img = cv2.imread('../input/flower.jpg')
    assert img is not None, "Failed to load image."
    
    pixels = np.reshape(img, (-1, 3))

    img_annotated = cv2.imread('../input/flower-petals-annotated.jpg')
    mask = cv2.inRange(img_annotated, (0, 0, 245), (10, 10, 256))
    mask_pixels = np.reshape(mask, (-1))
    cv2.imwrite('../output/ex10_annotation_mask.jpg', mask)

    # Determine mean value, standard deviations and covariance matrix
    # for the annotated pixels.
    # Using cv2 to calculate mean and standard deviations
    mean, std = cv2.meanStdDev(img, mask = mask)
    print("Mean color values of the annotated pixels")
    print(mean)
    print("Standard deviation of color values of the annotated pixels")
    print(std)


main()
