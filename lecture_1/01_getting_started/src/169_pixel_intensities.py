import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/input/flower.jpg"
    output_path = "lecture_1/01_getting_started/output/169/"
    output_path_image = output_path + "flower_output_gray.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path_image, gray)
    
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.savefig(output_path + "histogram.png")
    
    # henriks solution
    values = img[:, :, 1]
    plt.hist(np.reshape(values, (-1)), bins = 255)
    plt.savefig(output_path + "histogram_2.png")