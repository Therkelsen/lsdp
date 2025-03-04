import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def calc_mahal_dist(x, mean, cov):
    return math.sqrt(np.linalg.matmul(np.linalg.matmul((x-mean), np.linalg.inv(cov)), np.transpose(x-mean)))


def save_annotations():
    """
    Finds pixels annotated in an image, and saves the corresponding pixel values in the non-annotated image.
    """
    img = cv2.imread("image.JPG")
    img_anno = cv2.imread("image_annotated.PNG")

    BLUE = [255,0,0]
    data_values = pd.DataFrame(columns=['R', 'G', 'B'])
    data_indexs = pd.DataFrame(columns=['row', 'col'])

    for row in range(len(img_anno)):
        print(row, "/", len(img_anno))
        for col in range (len(img_anno[row])):
            pixel_anno = img_anno[row][col]
            if np.array_equal(pixel_anno, BLUE):
                pixel_org = img[row][col]
                data_values.loc[len(data_values)] = pixel_org[2], pixel_org[1], pixel_org[0]
                data_indexs.loc[len(data_indexs)] = row, col

    data_values.to_csv("data_values.csv", sep=",", na_rep="", index=False)
    data_indexs.to_csv("data_indexs.csv", sep=",", na_rep="", index=False)


def visualize_mahala_dist(n_bins: int):
    """
    Visualises the Mahalanobis distance in the red-green plot of the pixel values
    """
    pixels = pd.read_csv("data_values.csv")
    pixels = pixels.to_numpy()
    pixels_rg = np.delete(pixels, 2, 1)
    mean_rg = np.mean(pixels_rg, 0)
    cov_rg = np.cov(pixels_rg, rowvar=False)

    print("mean:", mean_rg)
    print("covar:", cov_rg)
    print("shape:", pixels_rg.shape)

    bin_size = 256/n_bins
    rg_hist = np.zeros((n_bins,n_bins))

    # Create heatmap
    for i in range(len(pixels)):
        rg_hist[round(pixels[i][0]/bin_size)-1, round(pixels[i][1]/bin_size)-1] += 1
    
    # plot mean
    plt.plot(round(mean_rg[1]/bin_size)-1, round(mean_rg[0]/bin_size)-1, 'x')

    # plot mahalanobis in heatmap
    for i in range(256):
        for j in range(256):
            x = np.array([i,j])
            mahal_dist = calc_mahal_dist(x=x, mean=mean_rg, cov=cov_rg)
            if 1 < mahal_dist < 1.2:
                rg_hist[round(i/bin_size)-1][round(j/bin_size)-1] = 400

    # Plot heatmap
    plt.imshow(rg_hist, cmap='hot')
    plt.colorbar()
    plt.show()



def create_color_mask(img_file_path, mean, cov, threshold_higher_bound = 9, threshold_lower_bound = None) -> any:
    if threshold_lower_bound == None:
        threshold_lower_bound = 0
    
    img = cv2.imread(img_file_path)
    #mask = np.zeros(shape=(len(img), len(img[0])))
    mask = np.zeros(shape=img.shape)

    for row in range(len(img)):
        if row % 10 == 0:
            print(row, "/", len(img))
        for col in range(len(img[row])):
            x = img[row][col]
            mahal_dist = calc_mahal_dist(x=x, mean=mean, cov=cov)
            
            if threshold_lower_bound < mahal_dist and mahal_dist < threshold_higher_bound:
                mask[row,col] = 255

    return np.max(mask) - mask  # TODO: Why do i have to invert the mask?
    
    
def plot_mahalanobis_histogram(img_file_path, mean, cov):
    img = cv2.imread(img_file_path)
    distances = []

    for row in range(len(img)):
        for col in range(len(img[row])):
            x = img[row][col]
            mahal_dist = calc_mahal_dist(x=x, mean=mean, cov=cov)
            distances.append(mahal_dist)
    
    # Plot histogram of distances
    plt.hist(distances, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mahalanobis Distances')
    plt.show()



if __name__ == "__main__":
    img_file = "mini_project_1/src/image_small.jpg"


    #save_annotations()
    pixels_anno = pd.read_csv("mini_project_1/src/data_values.csv").to_numpy()

    cov = np.cov(pixels_anno, rowvar=False)
    mean = np.mean(pixels_anno, 0)

    print("covar:\n", cov)
    print("mean:", mean)

    #visualize_mahala_dist(n_bins=50)

    #plot_mahalanobis_histogram(img_file, mean, cov)
 
    if True: # create and plot mask
        mask = create_color_mask(img_file, mean, cov, threshold_higher_bound=12, threshold_lower_bound=10)

        cv2.imshow("mask", cv2.resize(mask, (600,400)))
        cv2.imshow("img", cv2.resize(cv2.imread(img_file), (600,400)))
        cv2.waitKey(0)

        cv2.imwrite("mask.PNG", mask)
    