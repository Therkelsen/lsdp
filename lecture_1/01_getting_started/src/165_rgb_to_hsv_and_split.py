import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/deer.jpg"
    output_path = "lecture_1/01_getting_started/output/165/"
    
    output_path_h = output_path + "deer_output_h.png"
    output_path_s = output_path + "deer_output_s.png"
    output_path_v = output_path + "deer_output_v.png"
    
    # create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h,s,v=cv2.split(hsv_img)
    
    cv2.imwrite(output_path_h, h)
    cv2.imwrite(output_path_s, s)
    cv2.imwrite(output_path_v, v)