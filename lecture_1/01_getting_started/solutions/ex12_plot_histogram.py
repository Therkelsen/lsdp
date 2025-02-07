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
  print("Maximum color distance in image")
  print(np.max(euclidean_dist_image))

  # Plot the distribution of Euclidean distances in a histogram
  plt.hist(euclidean_dist_image.ravel(), 315, [0, 315]); 
  plt.title("Euclidean distance histogram")
  plt.savefig("../output/ex12_euclidean_distance_histogram.png")
  plt.close()
  plt.hist(euclidean_dist_image.ravel(), 256, [0, 256], log = True); 
  plt.title("Euclidean distance histogram - logarithmic")
  plt.savefig("../output/ex12_euclidean_distance_histogram_logarithmic.png")
  plt.close()


main()


