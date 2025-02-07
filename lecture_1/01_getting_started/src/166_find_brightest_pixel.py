import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/deer.jpg"
    output_path = "lecture_1/01_getting_started/output/166/"
    
    output_path_image = output_path + "deer_output_brightest_pixel.png"
    
    # create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    print(maxLoc)
    
    output_image = img.copy()
    
    cv2.circle(output_image, maxLoc, 5, (255, 0, 0), 2)
    
    cv2.imwrite(output_path_image, output_image)