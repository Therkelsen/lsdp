import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/flower.jpg"
    output_path = "lecture_1/01_getting_started/output/167/"
    
    output_path_mask_rgb = output_path + "flower_output_yellow_mask_rgb.png"
    output_path_mask_hsv = output_path + "flower_output_yellow_mask_hsv.png"
    output_path_mask_lab = output_path + "flower_output_yellow_mask_lab.png"
    output_path_image_rgb = output_path + "flower_output_yellow_image_rgb.png"
    output_path_image_hsv = output_path + "flower_output_yellow_image_hsv.png"
    output_path_image_lab = output_path + "flower_output_yellow_image_lab.png"
    
    # create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    lower_yellow_rgb = (0, 100, 100)
    upper_yellow_rgb = (100, 255, 255)
    yellow_mask_rgb = cv2.inRange(img, lower_yellow_rgb, upper_yellow_rgb)
    output_image_rgb = cv2.bitwise_and(img, img, mask=yellow_mask_rgb)
    cv2.imwrite(output_path_mask_rgb, yellow_mask_rgb)
    cv2.imwrite(output_path_image_rgb, output_image_rgb)    
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    
    lower_yellow_hsv = np.array([25/2, 150, 0])
    upper_yellow_hsv = np.array([50/2, 255, 255])
    yellow_mask_hsv = cv2.inRange(hsv_img, lower_yellow_hsv, upper_yellow_hsv)
    output_image_hsv = cv2.bitwise_and(img, img, mask=yellow_mask_hsv)
    cv2.imwrite(output_path_mask_hsv, yellow_mask_hsv)
    cv2.imwrite(output_path_image_hsv, output_image_hsv)
    
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lower_yellow_lab = (100, 0, 170)
    upper_yellow_lab = (255, 255, 255)
    yellow_mask_lab = cv2.inRange(lab_img, lower_yellow_lab, upper_yellow_lab)
    output_image_lab = cv2.bitwise_and(img, img, mask=yellow_mask_lab)
    cv2.imwrite(output_path_mask_lab, yellow_mask_lab)
    cv2.imwrite(output_path_image_lab, output_image_lab)