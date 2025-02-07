import cv2
from icecream import ic

img = cv2.imread("../input/deer.jpg")
assert img is not None, "Failed to load image."

R = img[:, :, 2]
G = img[:, :, 1]
B = img[:, :, 0]
cv2.imwrite("../output/ex03_red.jpg", R)
cv2.imwrite("../output/ex03_green.jpg", G)
cv2.imwrite("../output/ex03_blue.jpg", B)
