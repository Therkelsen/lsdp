import numpy as np
import cv2

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/deer.jpg"
    output_path = "lecture_1/01_getting_started/output/164/deer_output_half.png"
    
    img = cv2.imread(input_path)
    top_half_cropped_image = img[:img.shape[0]//2, :]
    cv2.imwrite(output_path, top_half_cropped_image)