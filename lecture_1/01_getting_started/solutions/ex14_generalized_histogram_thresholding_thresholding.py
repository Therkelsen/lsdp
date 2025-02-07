import numpy as np
import cv2
from matplotlib import pyplot as plt

# Code borrowed from:
# A Generalization of Otsu's Method and Minimum Error Thresholding by Jonathan T. Barron
# https://arxiv.org/abs/2007.07350
csum = lambda z : np.cumsum (z)[:-1]
dsum = lambda z : np.cumsum (z[::-1])[-2::-1]
argmax = lambda x, f : np.mean (x[: -1][f == np.max (f)])
clip = lambda z : np.maximum (1e-30, z)

# Use the mean for ties .
def preliminaries(n, x):
  """Some math that is shared across each algorithm."""
  assert np.all(n >= 0)
  x = np.arange(len (n), dtype = n.dtype) if x is None else x
  assert np.all(x[1:] >= x[: -1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum (n * x) / w0
  mu1 = dsum (n * x) / w1
  d0 = csum (n * x **2) - w0 * mu0 **2
  d1 = dsum (n * x **2) - w1 * mu1 **2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
  """Our generalization of the above algorithms."""
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
  v0 = clip((p0 * nu * tau **2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau **2 + d1) / (p1 * nu + w1))
  f0 = - d0 / v0 - w0 * np.log (v0) + 2 * (w0 + kappa * omega) * np.log (w0)
  f1 = - d1 / v1 - w1 * np.log (v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1



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

  hist_n, hist_edge = np.histogram(euclidean_dist_image_scaled, np.arange(-0.5, 256))
  hist_x = (hist_edge[1:] + hist_edge[:-1]) / 2.

  threshold_GHT, valb = GHT(hist_n, hist_x, nu=2**5, tau=2**10, kappa=0.1, omega=0.5)
  print("Threshold value found by Generalized Histogram Thresholding")
  print(threshold_GHT)

  thresholded_image = (euclidean_dist_image_scaled < threshold_GHT) * 255
  cv2.imwrite("../output/ex14_generalized_histogram_thresholding.jpg", thresholded_image)


main()


