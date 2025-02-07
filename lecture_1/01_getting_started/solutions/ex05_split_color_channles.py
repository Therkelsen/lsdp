import cv2

img = cv2.imread("../input/deer.jpg")
assert img is not None, "Failed to load image."

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H = img[:, :, 0]
S = img[:, :, 1]
V = img[:, :, 2]
cv2.imwrite("../output/ex05_hue.jpg", H)
cv2.imwrite("../output/ex05_saturation.jpg", S)
cv2.imwrite("../output/ex05_value.jpg", V)
