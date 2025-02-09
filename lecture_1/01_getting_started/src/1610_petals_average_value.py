import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path_1 = "lecture_1/01_getting_started/input/flower.jpg"
    input_path_2 = "lecture_1/01_getting_started/input/flower-petals-annotated.jpg"
    output_path = "lecture_1/01_getting_started/output/1610/"
    output_path_image = output_path + "flower_output_mask.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path_1)
    img_annotated = cv2.imread(input_path_2)
    
    mask = cv2.inRange(img_annotated, (0, 0, 245), (10, 10, 256))
    
    cv2.imwrite(output_path_image, mask)
    
    mean, std = cv2.meanStdDev(img, mask = mask)
    print("Mean color values of the annotated pixels")
    print(mean)
    print("Standard deviation of color values of the annotated pixels")
    print(std)