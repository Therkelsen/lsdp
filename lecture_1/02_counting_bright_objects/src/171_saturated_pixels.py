import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

if __name__ == "__main__":
    input_path = "lecture_1/02_counting_bright_objects/input/"
    # output_path = "lecture_1/01_getting_started/output/169/"
    # output_path_image = output_path + "flower_output_gray.png"
    
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # for all images in the input_path dir, save the image and the file name
    images = []
    file_names = []
    for file in os.listdir(input_path):
        if file.endswith(".JPG"):
            img = cv.imread(input_path + file)
            images.append(img)
            file_names.append(file)
            print('Loaded image:', file)
    
    # for all images in the images list,
    for img, file in zip(images, file_names):
        print("\nAnalysing file:", file)
        print("\tShape:", img.shape)
        print("\tPixels:", img.shape[0] * img.shape[1])
        saturated_pixels_mask = cv.inRange(img, (255, 255, 255), (255, 255, 255))
        saturated_pixels_count = np.count_nonzero(saturated_pixels_mask)
        print("\tFully saturated pixels: ", saturated_pixels_count)

        red_saturated_pixels_mask = cv.inRange(img, (0, 0, 255), (255, 255, 255))
        red_saturated_pixels_count = np.count_nonzero(red_saturated_pixels_mask)
        print("\tRed saturated pixels: ", red_saturated_pixels_count)

        blue_saturated_pixels_mask = cv.inRange(img, (255, 0, 0), (255, 255, 255))
        blue_saturated_pixels_count = np.count_nonzero(blue_saturated_pixels_mask)
        print("\tBlue saturated pixels: ", blue_saturated_pixels_count)

        green_saturated_pixels_mask = cv.inRange(img, (0, 255, 0), (255, 255, 255))
        green_saturated_pixels_count = np.count_nonzero(green_saturated_pixels_mask)
        print("\tGreen saturated pixels: ", green_saturated_pixels_count)

        partially_saturated = np.logical_or(red_saturated_pixels_mask, blue_saturated_pixels_mask, green_saturated_pixels_mask)
        partially_saturated_count = np.count_nonzero(partially_saturated)
        print("\tPartially saturated pixels:", partially_saturated_count)