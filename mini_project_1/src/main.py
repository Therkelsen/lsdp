import os
import math
import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.windows import Window

# def calc_mahal_dist(x, mean, cov):
#     return math.sqrt(np.linalg.matmul(np.linalg.matmul((x-mean), np.linalg.inv(cov)), np.transpose(x-mean)))


# def save_annotations():
#     """
#     Finds pixels annotated in an image, and saves the corresponding pixel values in the non-annotated image.
#     """
#     img = cv.imread("mini_project_1/inputs/image.JPG")
#     img_anno = cv.imread("mini_project_1/inputs/image_annotated.PNG")

#     BLUE = [255,0,0]
#     data_values = pd.DataFrame(columns=['R', 'G', 'B'])
#     data_indexs = pd.DataFrame(columns=['row', 'col'])

#     for row in range(len(img_anno)):
#         print(row, "/", len(img_anno))
#         for col in range (len(img_anno[row])):
#             pixel_anno = img_anno[row][col]
#             if np.array_equal(pixel_anno, BLUE):
#                 pixel_org = img[row][col]
#                 data_values.loc[len(data_values)] = pixel_org[2], pixel_org[1], pixel_org[0]
#                 data_indexs.loc[len(data_indexs)] = row, col

#     data_values.to_csv("mini_project_1/outputs/data_values.csv", sep=",", na_rep="", index=False)
#     data_indexs.to_csv("mini_project_1/outputs/data_indices.csv", sep=",", na_rep="", index=False)


# def visualize_mahala_dist(n_bins: int):
#     """
#     Visualises the Mahalanobis distance in the red-green plot of the pixel values
#     """
#     pixels = pd.read_csv("mini_project_1/outputs/data_values.csv")
#     pixels = pixels.to_numpy()
#     pixels_rg = np.delete(pixels, 2, 1)
#     mean_rg = np.mean(pixels_rg, 0)
#     cov_rg = np.cov(pixels_rg, rowvar=False)

#     print("mean:", mean_rg)
#     print("covar:", cov_rg)
#     print("shape:", pixels_rg.shape)

#     bin_size = 256/n_bins
#     rg_hist = np.zeros((n_bins,n_bins))

#     # Create heatmap
#     for i in range(len(pixels)):
#         rg_hist[round(pixels[i][0]/bin_size)-1, round(pixels[i][1]/bin_size)-1] += 1
    
#     # plot mean
#     plt.plot(round(mean_rg[1]/bin_size)-1, round(mean_rg[0]/bin_size)-1, 'x')

#     # plot mahalanobis in heatmap
#     for i in range(256):
#         for j in range(256):
#             x = np.array([i,j])
#             mahal_dist = calc_mahal_dist(x=x, mean=mean_rg, cov=cov_rg)
#             if 1 < mahal_dist < 1.2:
#                 rg_hist[round(i/bin_size)-1][round(j/bin_size)-1] = 400

#     # Plot heatmap
#     plt.imshow(rg_hist, cmap='hot')
#     plt.colorbar()
#     plt.show()



# def create_color_mask(img_file_path, mean, cov, threshold_higher_bound = 9, threshold_lower_bound = None) -> any:
#     if threshold_lower_bound == None:
#         threshold_lower_bound = 0
    
#     img = cv.imread(img_file_path)
#     #mask = np.zeros(shape=(len(img), len(img[0])))
#     mask = np.zeros(shape=img.shape)

#     for row in range(len(img)):
#         if row % 10 == 0:
#             print(row, "/", len(img))
#         for col in range(len(img[row])):
#             x = img[row][col]
#             mahal_dist = calc_mahal_dist(x=x, mean=mean, cov=cov)
            
#             if threshold_lower_bound < mahal_dist and mahal_dist < threshold_higher_bound:
#                 mask[row,col] = 255

#     return np.max(mask) - mask  # TODO: Why do i have to invert the mask?
    
    
# def plot_mahalanobis_histogram(img_file_path, mean, cov):
#     img = cv.imread(img_file_path)
#     distances = []

#     for row in range(len(img)):
#         for col in range(len(img[row])):
#             x = img[row][col]
#             mahal_dist = calc_mahal_dist(x=x, mean=mean, cov=cov)
#             distances.append(mahal_dist)
    
#     # Plot histogram of distances
#     plt.hist(distances, bins=50, color='blue', alpha=0.7)
#     plt.xlabel('Mahalanobis Distance')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Mahalanobis Distances')
#     plt.show()

def get_tiles(orthomosaic, output_path):
    columns, rows = orthomosaic.width, orthomosaic.height
    tile_size = 512
    
    print(f'Orthomosaic shape: {rows}x{columns}')
    print(f'Tile size: {tile_size}x{tile_size}')
    tiles_x = int(np.ceil(rows/tile_size))
    tiles_y = int(np.ceil(columns/tile_size))
    num_tiles = tiles_x * tiles_y
    print(f'Number of tiles: {tiles_x}x{tiles_y} = {num_tiles}')
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    tiles = 0
    for tile_y in range(0, rows, tile_size):
        for tile_x in range(0, columns, tile_size):
            lrc_x = min(tile_x + tile_size, columns)
            lrc_y = min(tile_y + tile_size, rows)

            window_location = Window(tile_x, tile_y, lrc_x - tile_x, lrc_y - tile_y)
            img = orthomosaic.read(window=window_location)
            
            temp = img.transpose(1, 2, 0)
            t2 = cv.split(temp)
            img_cv = cv.merge([t2[2], t2[1], t2[0]])
            # printing uses a lot of cpu time so it is commented out
            # print(f'Loaded tile at ({tile_x}, {tile_y}) with shape: {img_cv.shape}')
            
            tiles += 1
            # save tile as png with its tile index (tile_0, tile_1, etc.)
            cv.imwrite(f'{output_path}/tile_{tiles}.png', img_cv)
    return

def get_annotated_color_data(orig_img, anno_img, anno_color):
    mask = cv.inRange(anno_img, anno_color, anno_color)
    
    pixels_selected_by_anno = orig_img[mask > 0]
    
    mean = np.mean(pixels_selected_by_anno, axis=0)
    std = np.std(pixels_selected_by_anno, axis=0)
    
    return mask, pixels_selected_by_anno, mean, std


def plot_color_distribution(image_masked_no_zeros, color_space):
    """Plot the histograms of the RGB color channels."""
    
    # Split the channels
    channels = []
    colors = []
    labels = []
    if color_space == "RGB":
        r_channel = image_masked_no_zeros[:, 2]  # Red channel
        g_channel = image_masked_no_zeros[:, 1]  # Green channel
        b_channel = image_masked_no_zeros[:, 0]  # Blue channel
        channels = [r_channel, g_channel, b_channel]
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
    elif color_space == "Lab":
        l_channel = image_masked_no_zeros[:, 0]  # L channel
        a_channel = image_masked_no_zeros[:, 1]  # a channel
        b_channel = image_masked_no_zeros[:, 2]  # b channel
        channels = [l_channel, a_channel, b_channel]
        colors = ['gray', 'purple', 'orange']
        labels = ['L', 'a', 'b']
    else:
        raise ValueError("Invalid color space. Use 'RGB' or 'Lab'.")

    # Create the histogram
    plt.figure(figsize=(10, 5))

    # Plot histograms for each channel
    for channel, color, label in zip(channels, colors, labels):
        hist, bins = np.histogram(channel.ravel(), bins=256)
        max_bin_index = np.argmax(hist)
        hist[max_bin_index] = 0  # Remove the bin with the highest frequency
        plt.bar(bins[:-1], hist, color=color, alpha=0.6, label=label, width=1)

    # Labels and title
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(color_space + ' Color Histogram')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # orthomosaic_path = "mini_project_1/inputs/segmented_orthomosaic.tif"
    # output_path = "mini_project_1/outputs"

    # with rio.open(orthomosaic_path) as src:
    #     tiles = get_tiles(src, output_path)
    #     print (f'Loaded {tiles} tiles from {orthomosaic_path}')

    orig_img_bgr = cv.imread("mini_project_1/inputs/image.JPG")
    orig_img_rgb = cv.cvtColor(orig_img_bgr, cv.COLOR_BGR2RGB)
    orig_img_lab = cv.cvtColor(orig_img_bgr, cv.COLOR_BGR2Lab)
    
    anno_img_bgr = cv.imread("mini_project_1/inputs/image_annotated.PNG")
    anno_img_rgb = cv.cvtColor(anno_img_bgr, cv.COLOR_BGR2RGB)
    anno_img_lab = cv.cvtColor(anno_img_bgr, cv.COLOR_BGR2Lab)
    
    anno_color_rgb = np.array([0, 0, 255], dtype=np.uint8)
    
    anno_color_bgr = anno_color_rgb[::-1]  # Convert RGB → BGR
    anno_color_lab = cv.cvtColor(anno_color_bgr.reshape(1, 1, 3), cv.COLOR_BGR2Lab).reshape(3)
    anno_color_lab = anno_color_lab.astype(np.uint8)  # Convert to uint8 for OpenCV
    
    # Get annotated color data
    pumpkin_color_data_rgb = get_annotated_color_data(orig_img_rgb, anno_img_rgb, anno_color_rgb)
    pumpkin_color_data_lab = get_annotated_color_data(orig_img_lab, anno_img_lab, anno_color_lab)
    
    # Print out the RGB and Lab means and standard deviations
    print("\nPumpkin color (RGB):\nMean:", pumpkin_color_data_rgb[2], "\nStd:", pumpkin_color_data_rgb[3])
    print("\nPumpkin color (Lab):\nMean:", pumpkin_color_data_lab[2], "\nStd:", pumpkin_color_data_lab[3])
    
    # Plot the color distributions for the pixels selected by annotation
    plot_color_distribution(pumpkin_color_data_rgb[1], 'RGB')
    plot_color_distribution(pumpkin_color_data_lab[1], 'Lab')
    # plot_color_distribution(pumpkin_color_data_lab[1], color_space="Lab")
    
    # curr_img = "mini_project_1/outputs/tiles/tile_1.png"
    
    # img_file = "mini_project_1/inputs/image_small.jpg"

    # #save_annotations()
    # pixels_anno = pd.read_csv("mini_project_1/outputs/data_values.csv").to_numpy()

    # cov = np.cov(pixels_anno, rowvar=False)
    # mean = np.mean(pixels_anno, 0)

    # print("covar:\n", cov)
    # print("mean:", mean)

    # #visualize_mahala_dist(n_bins=50)

    # #plot_mahalanobis_histogram(img_file, mean, cov)
 
    # if True: # create and plot mask
    #     mask = create_color_mask(img_file, mean, cov, threshold_higher_bound=12, threshold_lower_bound=10)
    #     mask = cv.normalize(mask, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

    #     cv.imshow("mask", cv.resize(mask, (600,400)))
    #     cv.imshow("img", cv.resize(cv.imread(img_file), (600,400)))
    #     cv.waitKey(0)

    #     cv.imwrite("mini_project_1/outputs/mask.PNG", mask)