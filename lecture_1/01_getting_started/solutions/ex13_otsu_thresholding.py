import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
  img = cv2.imread('../input/flower.jpg')
  assert img is not None, "Failed to load image."
  
  pixels = np.reshape(img, (-1, 3))
  reference_color = [187, 180, 170] # Given in BGR

  # Calculate the euclidean distance to the reference_color annotated color.
  shape = pixels.shape
  diff = pixels - np.repeat([reference_color], shape[0], axis=0)
  euclidean_dist = np.sqrt(np.sum(diff * diff, axis=1))
  euclidean_dist_image = np.reshape(euclidean_dist, 
          (img.shape[0], img.shape[1]))

  euclidean_dist_image_scaled = 255 * euclidean_dist_image / np.max(euclidean_dist_image)
  threshold_value, segmented_image = cv2.threshold(
          euclidean_dist_image_scaled.astype(np.uint8), 
          0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  print("Threshold value found by Otsu's method")
  print(threshold_value)
  cv2.imwrite("../output/ex13_euclidean_dist_image_thresholded_otsu.jpg",
              segmented_image)


main()


