import numpy as np
import cv2

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/deer.jpg"
    r_output_path = "lecture_1/01_getting_started/output/163/deer_output_red_channel.png"
    g_output_path = "lecture_1/01_getting_started/output/163/deer_output_green_channel.png"
    b_output_path = "lecture_1/01_getting_started/output/163/deer_output_blue_channel.png"
    
    img = cv2.imread(input_path)
    b,g,r=cv2.split(img)
    
    cv2.imwrite(r_output_path, r)
    cv2.imwrite(g_output_path, g)
    cv2.imwrite(b_output_path, b)