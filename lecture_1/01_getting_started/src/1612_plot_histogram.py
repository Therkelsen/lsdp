import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path_euclidean = "lecture_1/01_getting_started/output/1611/flower_euclidean_distances_image.png"
    input_path_mahalanobis = "lecture_1/01_getting_started/output/1611/flower_mahalanobis_distances_image.png"    
    output_path = "lecture_1/01_getting_started/output/1612/"
    output_path_euclidean_histogram = output_path + "flower_euclidean_distances_histogram.png"
    output_path_mahalanobis_histogram = output_path + "flower_mahalanobis_distances_histogram.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img_euclidean = cv2.imread(input_path_euclidean)
    img_mahalanobis = cv2.imread(input_path_mahalanobis)
    
    # plot histogram and save it as an image
    plt.hist(img_euclidean.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
    plt.savefig(output_path_euclidean_histogram)
    plt.clf()
    
    plt.hist(img_mahalanobis.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
    plt.savefig(output_path_mahalanobis_histogram)
    plt.clf()
    
    print("Histograms saved successfully.")