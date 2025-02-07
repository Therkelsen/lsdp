import cv2

img = cv2.imread("../input/deer.jpg")
assert img is not None, "Failed to load image."

cropped_img = img[0:200, :, :]
cv2.imwrite("../output/ex04_upperhalf.jpg", cropped_img)
