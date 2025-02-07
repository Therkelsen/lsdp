import cv2

img = cv2.imread('../input/deer.jpg')
assert img is not None, "Failed to load image."

(minval, maxval, minloc, maxloc) = cv2.minMaxLoc(img[:, :, 1])
cv2.circle(img, maxloc, 20, (0, 255, 255), 3)
cv2.imwrite("../output/ex06_max_intensity_located.jpg", img)


