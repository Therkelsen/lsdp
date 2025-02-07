import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():
  filename = "../input/well_exposed_DJI_0214.JPG"
  img = cv2.imread(filename)
  plt.hist(img.ravel(), 256, [0, 256]); 
  plt.title("Well exposed histogram")
  plt.savefig("../output/ex11_well_exposed_histogram.png")
  plt.close()
  plt.hist(img.ravel(), 256, [0, 256], log = True); 
  plt.title("Well exposed histogram - logarithmic")
  plt.savefig("../output/ex11_well_exposed_histogram_logarithmic.png")
  plt.close()

  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  plt.hist(img.ravel(), 256, [0, 256]); 
  plt.savefig("../output/ex11_well_exposed_histogram_grayscale.png")
  plt.close()
  plt.hist(img.ravel(), 256, [0, 256], log = True); 
  plt.savefig("../output/ex11_well_exposed_histogram_grayscale_logarithmic.png")
  plt.close()

  filename = "../input/under_exposed_DJI_0213.JPG"
  img = cv2.imread(filename)
  plt.hist(img.ravel(), 256, [0, 256]); 
  plt.title("Under exposed histogram")
  plt.savefig("../output/ex11_under_exposed_histogram.png")
  plt.close()
  plt.hist(img.ravel(), 256, [0, 256], log = True); 
  plt.title("Under exposed histogram - logarithmic")
  plt.savefig("../output/ex11_under_exposed_histogram_logarithmic.png")
  plt.close()

  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  plt.hist(img.ravel(), 256, [0, 256]); 
  plt.savefig("../output/ex11_under_exposed_histogram_grayscale.png")
  plt.close()
  plt.hist(img.ravel(), 256, [0, 256], log = True); 
  plt.savefig("../output/ex11_under_exposed_histogram_grayscale_logarithmic.png")
  plt.close()



main()


