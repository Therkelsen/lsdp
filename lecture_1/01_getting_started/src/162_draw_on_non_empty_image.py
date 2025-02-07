import numpy as np
import cv2

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/deer.jpg"
    output_path = "lecture_1/01_getting_started/output/162/deer_output.png"
    
    img = cv2.imread(input_path)
    cv2.line(img, (20, 30), (40, 120),(0, 0, 255), 3)
    cv2.imwrite(output_path, img)