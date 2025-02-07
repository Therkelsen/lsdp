import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/flower.jpg"
    output_path = "lecture_1/01_getting_started/output/167/"
    
    output_path_image = output_path + "flower_output_yellow_mask.png"
    
    # create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([25/2, 150, 0])
    upper_yellow = np.array([50/2, 255, 255])
    
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    
    output_image = cv2.bitwise_and(img, img, mask=yellow_mask)
    
    cv2.imwrite(output_path_image, output_image)