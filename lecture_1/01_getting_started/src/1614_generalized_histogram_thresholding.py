import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

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

if __name__ == "__main__":
    input_path_euclidean = "lecture_1/01_getting_started/output/1611/flower_euclidean_distances_image.png"
    input_path_mahalanobis = "lecture_1/01_getting_started/output/1611/flower_mahalanobis_distances_image.png"    
    output_path = "lecture_1/01_getting_started/output/1614/"
    output_path_euclidean_ght = output_path + "flower_euclidean_ght_thresholded.png"
    output_path_mahalanobis_ght = output_path + "flower_mahalanobis_ght_thresholded.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the images in grayscale mode
    img_euclidean = cv2.imread(input_path_euclidean)
    img_mahalanobis = cv2.imread(input_path_mahalanobis)
    
    # Use the Generalized Histogram Thresholding method for determining the threshold to segment the distance images
    # Euclidean distance image
    hist_n, hist_edge = np.histogram(img_euclidean, np.arange(-0.5, 256))
    hist_x = (hist_edge[1:] + hist_edge[:-1]) / 2.

    threshold_GHT, valb = GHT(hist_n, hist_x, nu=2**5, tau=2**10, kappa=0.1, omega=0.5)
    print("Euclidean threshold value found by Generalized Histogram Thresholding")
    print(threshold_GHT)

    thresholded_image = ((img_euclidean < threshold_GHT) * 255).astype(np.uint8)
    cv2.imwrite(output_path_euclidean_ght, thresholded_image)
    
    # Mahalanobis distance image
    hist_n, hist_edge = np.histogram(img_mahalanobis, np.arange(-0.5, 256))
    hist_x = (hist_edge[1:] + hist_edge[:-1]) / 2.
    
    threshold_GHT, valb = GHT(hist_n, hist_x, nu=2**5, tau=2**10, kappa=0.1, omega=0.5)
    print("Euclidean threshold value found by Generalized Histogram Thresholding")
    print(threshold_GHT)
    
    thresholded_image = ((img_mahalanobis < threshold_GHT) * 255).astype(np.uint8)
    cv2.imwrite(output_path_mahalanobis_ght, thresholded_image)