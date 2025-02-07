import numpy as np
import cv2

if __name__ == "__main__":
    img = np.zeros((100, 200, 3), np.uint8)
    cv2.line(img, (20, 30), (40, 120),(0, 0, 255), 3)
    cv2.imwrite("lecture_1/01_getting_started/output/161/test.png", img)