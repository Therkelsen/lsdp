import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
  img = cv2.imread('../input/flower.jpg')
  assert img is not None, "Failed to load image."
  
  pixels = np.reshape(img, (-1, 3))
  reference_color = [187, 180, 170] # Given in BGR
  covariance_matrix = [[611, 684, 708], [684, 842, 888], [708, 888, 948]]

  # Calculate the euclidean distance to the reference_color annotated color.
  shape = pixels.shape
  diff = pixels - np.repeat([reference_color], shape[0], axis=0)
  inv_cov = np.linalg.inv(covariance_matrix)
  moddotproduct = diff * (diff @ inv_cov)
  mahalanobis_dist = np.sum(moddotproduct, 
      axis=1)
  mahalanobis_distance_image = np.reshape(
      mahalanobis_dist, 
        (img.shape[0],
         img.shape[1]))

  # Scale the distance image and export it.
  mahalanobis_distance_image = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
  cv2.imwrite("../output/ex11_mahalanobis_dist_image.jpg",
          mahalanobis_distance_image)


main()


