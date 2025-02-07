import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/flower.jpg"
    output_path = "lecture_1/01_getting_started/output/168/"
    
    output_path_image = output_path + "flower_output_rgb_intensities_row_plot.png"
    
    #create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    
    selected_line = 100
    green_values = img[selected_line, :, 1]
    blue_values = img[selected_line, :, 2]
    red_values = img[selected_line, :, 0]
    
    plt.plot(green_values, color = "green")
    plt.plot(blue_values, color = "blue")
    plt.plot(red_values, color = "red")
    
    plt.savefig(output_path + "rgb_intensities_row_plot.png")
    
    img[selected_line, :, :] = 255
    cv2.imwrite(output_path_image, img)