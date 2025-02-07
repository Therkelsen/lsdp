import cv2

img = cv2.imread("../input/flower.jpg")
assert img is not None, "Failed to load image."

lower_limit = (0, 100, 100)
upper_limit = (100, 255, 255)
mask = cv2.inRange(img, lower_limit, upper_limit)
cv2.imwrite("../output/ex07_yellowpart_rgb.jpg", mask)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(img_hsv, (10, 0, 100), (50, 255, 255))
lower_limit = (10, 0, 100)
upper_limit = (50, 255, 255)
mask = cv2.inRange(img_hsv, lower_limit, upper_limit)
cv2.imwrite("../output/ex07_yellowpart_hsv.jpg", mask)

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#mask = cv2.inRange(img_lab, (100, 0, 170), (255, 255, 255))
lower_limit = (100, 0, 170)
upper_limit = (255, 255, 255)
mask = cv2.inRange(img_lab, lower_limit, upper_limit)
cv2.imwrite("../output/ex07_yellowpart_lab.jpg", mask)

