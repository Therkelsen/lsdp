import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load image and extract values from it
img = cv2.imread("../input/flower.jpg")
assert img is not None, "Failed to load image."

values = img[:, :, 1]

plt.hist(np.reshape(values, (-1)), bins = 255)
plt.savefig("../output/ex09_histogram.png")


