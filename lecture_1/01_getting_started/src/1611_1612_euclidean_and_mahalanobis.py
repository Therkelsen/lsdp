import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

if __name__ == "__main__":
    input_path = "lecture_1/01_getting_started/output/167/flower_output_yellow_image_hsv.png"
    output_path = "lecture_1/01_getting_started/output/1611/"
    output_path_graph_euclidean = output_path + "flower_euclidean_distances.png"
    output_path_graph_mahalanobis = output_path + "flower_mahalanobis_distances.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)

    reference_color = np.array([178, 180, 187], dtype=np.float32)  # Ensure it's float32
    distances_euclidean = []
    distances_mahalanobis = []
    
    pixel_values = img.reshape(-1, 3).astype(np.float32)  # Flatten image into a list of RGB values
    cov_matrix = np.cov(pixel_values.T)  # Covariance of color channels
    inv_cov_matrix = np.linalg.inv(cov_matrix).astype(np.float32)  # Inverse of covariance matrix
    
    print("Distances between pixels and the reference color")

    rows, cols, _ = img.shape  # Get image dimensions

    for i in range(rows):
        for j in range(cols):
            # Grab the color at the current pixel
            comparison_color = img[i, j]
            comparison_color = comparison_color.astype(np.float32)  # Convert to float32
            # Euclidean distance
            distance_euclidean = cv2.norm(reference_color, comparison_color, cv2.NORM_L2)
            distances_euclidean.append(distance_euclidean)

            # Mahalanobis distance
            distance_mahalanobis = cv2.Mahalanobis(reference_color, comparison_color, inv_cov_matrix)
            distances_mahalanobis.append(distance_mahalanobis)
            
    mean_euclidean = np.mean(distances_euclidean)
    std_euclidean = np.std(distances_euclidean)
    print(f"Mean distance: {mean_euclidean}")
    print(f"Standard deviation of distances: {std_euclidean}")
    plt.hist(distances_euclidean, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(mean_euclidean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_euclidean:.2f}")
    plt.axvline(mean_euclidean + std_euclidean, color='green', linestyle='dotted', linewidth=2, label=f"Mean + Std: {mean_euclidean + std_euclidean:.2f}")
    plt.axvline(mean_euclidean - std_euclidean, color='green', linestyle='dotted', linewidth=2, label=f"Mean - Std: {mean_euclidean - std_euclidean:.2f}")
    plt.legend()
    plt.savefig(output_path_graph_euclidean)
    plt.clf()
    
    mean_mahalanobis = np.mean(distances_mahalanobis)
    std_mahalanobis = np.std(distances_mahalanobis)
    print(f"Mean distance: {mean_mahalanobis}")
    print(f"Standard deviation of distances: {std_mahalanobis}")
    plt.hist(distances_mahalanobis, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(mean_mahalanobis, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_mahalanobis:.2f}")
    plt.axvline(mean_mahalanobis + std_mahalanobis, color='green', linestyle='dotted', linewidth=2, label=f"Mean + Std: {mean_mahalanobis + std_mahalanobis:.2f}")
    plt.axvline(mean_mahalanobis - std_mahalanobis, color='green', linestyle='dotted', linewidth=2, label=f"Mean - Std: {mean_mahalanobis - std_mahalanobis:.2f}")
    plt.legend()
    plt.savefig(output_path_graph_mahalanobis)
