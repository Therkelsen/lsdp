import cv2
from icecream import ic

img = cv2.imread('../input/deer.jpg')
assert img is not None, "Failed to load image."

cv2.circle(img, (50, 10), 30, (255, 255, 0), 3)
cv2.imwrite("../output/ex02_draw_on_image.png", img)
