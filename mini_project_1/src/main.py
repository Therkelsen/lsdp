import os
import math
import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from rasterio.windows import Window

def num2str(num, digits=3):
    if num < 10:
        return f'00{num}'
    elif num < 100:
        return f'0{num}'
    else:
        return str(num)
    
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

    num_tiles = 0
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
            
            num_tiles += 1
            # save tile as png with its tile index (tile_0, tile_1, etc.)
            cv.imwrite(f'{output_path}/tile_{num2str(num_tiles)}.png', img_cv)
    return num_tiles

def load_tiles_lazy(tile_path):
    for filename in sorted(os.listdir(tile_path)):  # Sort to maintain order
        if filename.endswith('.png') or filename.endswith('.jpg'):  # Ensure it's an image
            file_path = os.path.join(tile_path, filename)
            img = cv.imread(file_path)  # Load only one tile at a time
            yield filename, img  # Yield filename and image instead of storing them all in memory

def is_image_empty(image):
    #return true if all pixels are the same color
    return np.all(image == image[0])

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
    if color_space == "BGR":
        r_channel = image_masked_no_zeros[:, 2]  # Red channel
        g_channel = image_masked_no_zeros[:, 1]  # Green channel
        b_channel = image_masked_no_zeros[:, 0]  # Blue channel
        channels = [r_channel, g_channel, b_channel]
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
    if color_space == "RGB":
        r_channel = image_masked_no_zeros[:, 0]  # Red channel
        g_channel = image_masked_no_zeros[:, 1]  # Green channel
        b_channel = image_masked_no_zeros[:, 2]  # Blue channel
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
    
def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)
    
def segment_image_by_color_inrange(image, color, tolerance_in_percent, color_space, method="inRange"):
    lower_bound = None
    upper_bound = None
    
    if color_space == "RGB":
        tolerance = 255 * tolerance_in_percent / 100
        lower_bound = color - tolerance
        upper_bound = color + tolerance
    elif color_space == "Lab":
        tolerance = (255 * tolerance_in_percent / 100)
        
        l_tolerance = tolerance * (100 / 255)  # L channel ranges from 0 to 100
        ab_tolerance = tolerance * (128 / 255)  # a/b channels range from -128 to 127
        
        lower_bound = np.array([color[0] - l_tolerance, color[1] - ab_tolerance, color[2] - ab_tolerance])
        upper_bound = np.array([color[0] + l_tolerance, color[1] + ab_tolerance, color[2] + ab_tolerance])
    else:
        raise ValueError("Invalid color space. Use 'RGB' or 'Lab'.")
    
    return cv.inRange(image, lower_bound, upper_bound)

def segment_image_by_color_distance(image, reference_color, distance_image, threshold, method="Euclidean"):
    pixels = np.reshape(image, (-1, 3))
    shape = pixels.shape
    if method == "Euclidean":
        # segment pumpkins in RGB space using Euclidean distance and Mahalanobis distance to reference color
        diff = pixels - np.repeat([reference_color], shape[0], axis=0)
        euclidean_dist = np.sqrt(np.sum(diff * diff, axis=1))
        euclidean_dist_image = np.reshape(euclidean_dist, 
                (image.shape[0], image.shape[1]))

        euclidean_dist_image_scaled = 255 * euclidean_dist_image / np.max(euclidean_dist_image)
        euclidean_dist_image_scaled = euclidean_dist_image_scaled.astype(np.uint8)
        
        segmented_image_euclidean = cv.inRange(euclidean_dist_image_scaled, threshold, 255)
        return np.max(segmented_image_euclidean) - segmented_image_euclidean
    elif method == "Mahalanobis":
        covariance_matrix = np.cov(pixels, rowvar=False)
        diff = pixels - np.repeat([reference_color], shape[0], axis=0)
        inv_cov = np.linalg.pinv(covariance_matrix)
        moddotproduct = diff * (diff @ inv_cov)
        mahalanobis_dist = np.sum(moddotproduct, 
            axis=1)
        mahalanobis_distance_image = np.reshape(
            mahalanobis_dist, 
                (image.shape[0],
                image.shape[1]))

        # Scale the distance image and export it.
        if np.max(mahalanobis_distance_image) > 0:
            mahalanobis_distance_image = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
        else:
            print("Warning: Mahalanobis distance is zero everywhere.")
        mahalanobis_distance_image = mahalanobis_distance_image.astype(np.uint8)
        segmented_image_mahalanobis = cv.inRange(mahalanobis_distance_image, threshold, 255)
        return np.max(segmented_image_mahalanobis) - segmented_image_mahalanobis
    else:
        raise ValueError("Invalid method. Use 'Euclidean' or 'Mahalanobis'.")
    
def create_segmented_image_for_report():
    path_image_orig = "mini_project_1/inputs/image.JPG"
    path_image_anno = "mini_project_1/inputs/image_annotated.PNG"
    path_image_smal = "mini_project_1/inputs/image_small.JPG"    
    
    # 3.1.1 - Calculate mean, std in RGB and Lab color spaces, visualize color distributions.
    orig_img_bgr = cv.imread(path_image_orig)
    orig_img_rgb = cv.cvtColor(orig_img_bgr, cv.COLOR_BGR2RGB)
    orig_img_lab = cv.cvtColor(orig_img_bgr, cv.COLOR_BGR2Lab)
    
    anno_img_bgr = cv.imread(path_image_anno)
    anno_img_rgb = cv.cvtColor(anno_img_bgr, cv.COLOR_BGR2RGB)
    anno_img_lab = cv.cvtColor(anno_img_bgr, cv.COLOR_BGR2Lab)
    
    anno_color_rgb = np.array([0, 0, 255], dtype=np.uint8)
    
    anno_color_bgr = anno_color_rgb[::-1]  # Convert RGB → BGR
    anno_color_lab = cv.cvtColor(anno_color_bgr.reshape(1, 1, 3), cv.COLOR_BGR2Lab).reshape(3)
    anno_color_lab = anno_color_lab.astype(np.uint8)  # Convert to uint8 for OpenCV
    
    # Get annotated color data
    pumpkin_color_data_rgb = get_annotated_color_data(orig_img_rgb, anno_img_rgb, anno_color_rgb)
    pumpkin_color_data_lab = get_annotated_color_data(orig_img_lab, anno_img_lab, anno_color_lab)

    # 3.1.2 - Segment pumpkins in RGB and Lab color spaces using inRange and distance in RGB space to reference color
    small_img_bgr = cv.imread(path_image_smal)
    small_img_rgb = cv.cvtColor(small_img_bgr, cv.COLOR_BGR2RGB)
    small_img_lab = cv.cvtColor(small_img_bgr, cv.COLOR_BGR2Lab)
    
    # Create segmented images
    segmented_image_rgb = segment_image_by_color_inrange(small_img_rgb, pumpkin_color_data_rgb[2], 15, "RGB")
    segmented_image_lab = segment_image_by_color_inrange(small_img_lab, pumpkin_color_data_lab[2], 15, "Lab")
    segmented_image_euclidean = segment_image_by_color_distance(small_img_rgb, pumpkin_color_data_rgb[2], None, 60, "Euclidean")
    segmented_image_mahalanobis = segment_image_by_color_distance(small_img_rgb, pumpkin_color_data_rgb[2], None, 35, "Mahalanobis")

    # Show 2x2 comparison image of segmentations
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(hspace=10, wspace=10)
    rows = 2
    cols = 2
    ax1 = plt.subplot(rows, cols, 1)
    ax1.imshow(segmented_image_rgb)
    ax1.set_title("RGB")
    ax1.axis('off')
    ax2 = plt.subplot(rows, cols, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented_image_lab)
    ax2.set_title("LAB")
    ax2.axis('off')
    ax3 = plt.subplot(rows, cols, 3, sharex=ax1, sharey=ax1)
    ax3.imshow(segmented_image_euclidean)
    ax3.set_title("Euclidian")
    ax3.axis('off')
    ax4 = plt.subplot(rows, cols, 4, sharex=ax1, sharey=ax1)
    ax4.imshow(segmented_image_mahalanobis)
    ax4.set_title("Mahalanobis")
    ax4.axis('off')
    plt.tight_layout()

    # Show origional image
    plt.figure()
    ax5 = plt.subplot(1,1,1, sharex=ax1, sharey=ax1)
    ax5.imshow(small_img_rgb)
    ax5.axis('off')
    plt.tight_layout()

    plt.show()

    # Save the four segmented images as square pdf's
    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1,1.3)                     # Expand upwards to include title 
    height = abs(extent.extents[2] - extent.extents[0])
    width = abs(extent.extents[1] - extent.extents[3])
    extent = extent.expanded(width/height, 1)           # Expand sideways to make square
    fig.savefig('ax1_figure_expanded.pdf', bbox_inches=extent)

    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1,1.3) 
    height = abs(extent.extents[2] - extent.extents[0])
    width = abs(extent.extents[1] - extent.extents[3])
    extent = extent.expanded(width/height, 1)
    fig.savefig('ax2_figure_expanded.pdf', bbox_inches=extent)

    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1,1.3)
    height = abs(extent.extents[2] - extent.extents[0])
    width = abs(extent.extents[1] - extent.extents[3])
    extent = extent.expanded(width/height, 1)
    fig.savefig('ax3_figure_expanded.pdf', bbox_inches=extent)

    extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1,1.3)
    height = abs(extent.extents[2] - extent.extents[0])
    width = abs(extent.extents[1] - extent.extents[3])
    extent = extent.expanded(width/height, 1)
    fig.savefig('ax4_figure_expanded.pdf', bbox_inches=extent)

def is_circle_near_edge(cx, cy, r, x0, y0, x1, y1):
    return (
        cx - r <= x0 or  # Near left edge
        cx + r >= x1 or  # Near right edge
        cy - r <= y0 or  # Near top edge
        cy + r >= y1     # Near bottom edge
    )
    
def percent(pop, sample):
    return int(pop/sample*100)

if __name__ == '__main__':
    print('Starting Pumpkin counting program.')
    
    path_image_orig = "mini_project_1/inputs/image.JPG"
    path_image_anno = "mini_project_1/inputs/image_annotated.PNG"
    path_image_smal = "mini_project_1/inputs/image_small.JPG"
    
    # 3.1.1 - Calculate mean, std in RGB and Lab color spaces, visualize color distributions.
    orig_img_bgr = cv.imread(path_image_orig)
    orig_img_rgb = cv.cvtColor(orig_img_bgr, cv.COLOR_BGR2RGB)
    orig_img_lab = cv.cvtColor(orig_img_bgr, cv.COLOR_BGR2Lab)
    
    anno_img_bgr = cv.imread(path_image_anno)
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
    #plot_color_distribution(pumpkin_color_data_rgb[1], 'RGB')
    #plot_color_distribution(pumpkin_color_data_lab[1], 'Lab')
    
    # 3.4.1 - 3.4.2 - Load orthomosaic and split into tiles
    orthomosaic_path = "mini_project_1/inputs/segmented_orthomosaic.tif"
    tile_path = "mini_project_1/outputs/tiles"

    with rio.open(orthomosaic_path) as src:
        num_tiles = get_tiles(src, tile_path)
        print (f'Loaded {num_tiles} tiles from {orthomosaic_path}')
        # print('\nPrinting some of the tile processing outputs:')
        num_empty = 0
        num_tiles = 0
        cum_sum_pump = 0
        cum_sum_edge = 0
        cum_sum_too_small = 0
        cum_sum_pump_area = []
        for filename, img in load_tiles_lazy(tile_path):
            if is_image_empty(img):
                # print(f"Skipping {filename} as it is empty")
                num_empty += 1
                continue
            num_tiles += 1
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            # print(f"Processing {filename} with shape {img.shape}:")  # Process one image at a time
            # plt.imsave('mini_project_1/outputs/input_img.png', img)
            # cv.imshow(filename, img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            
            # 3.4.3 - Count pumpkins in each tile

            # # 3.1.2 - Segment pumpkins in RGB and Lab color spaces using inRange and distance in RGB space to reference color
            # small_img_bgr = cv.imread(path_image_smal)
            # small_img_rgb = cv.cvtColor(small_img_bgr, cv.COLOR_BGR2RGB)
            # small_img_lab = cv.cvtColor(small_img_bgr, cv.COLOR_BGR2Lab)
            # tolerance = 10 #percent
            
            # segmented_image_rgb = segment_image_by_color_inrange(small_img_rgb, pumpkin_color_data_rgb[2], tolerance, "RGB")
            # # compare_original_and_segmented_image(small_img_rgb, segmented_image_rgb, "Segmented Pumpkins RGB")
            
            # segmented_image_lab = segment_image_by_color_inrange(small_img_lab, pumpkin_color_data_lab[2], tolerance, "Lab")
            # # compare_original_and_segmented_image(small_img_rgb, segmented_image_lab, "Segmented Pumpkins Lab")
            
            # # Euclidean segmentation
            # segmented_image_euclidean = segment_image_by_color_distance(small_img_rgb, pumpkin_color_data_rgb[2], None, 50, "Euclidean")
            # # compare_original_and_segmented_image(small_img_rgb, segmented_image_euclidean, "Segmented Pumpkins Euclidean RGB")
            
            # Mahalanobis segmentation
            segmented_image_mahalanobis = segment_image_by_color_distance(img, pumpkin_color_data_rgb[2], None, 40, "Mahalanobis")
            # cv.imshow("Segmented Pumpkins Mahalanobis", segmented_image_mahalanobis)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            
            # print(f'Segmented image shape: {segmented_image_mahalanobis.shape}')
            # compare_original_and_segmented_image(img, segmented_image_mahalanobis, "Segmented Pumpkins Mahalanobis")
            
            # # 3.2.1 - Count orange blobs
            # contours, hierarchy = cv.findContours(segmented_image_mahalanobis, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # print("Pumpkins found before opening:", len(contours))

            # 3.2.2 - Filter segmented image
            # Jeg forstår ikke hvorfor vi skal filtrere vores segmenterede billede. Det er jå binært...
            # Derfor laver jeg erosion i stedet for. 
            # Det burde minimere mængden bunker af græskar som bliver talt som ét græskar
            # og der burde fjerne enkelte punkter som er for små til at kunne vær et græskar
            kernel = np.ones((5,5), np.uint8)

            # run an opening on the segmented image
            # segmented_image_mahalanobis_filtered = cv.erode(segmented_image_mahalanobis, kernel=kernel)
            segmented_image_mahalanobis_filtered = cv.morphologyEx(segmented_image_mahalanobis, cv.MORPH_OPEN, kernel)
            # plt.imsave('mini_project_1/outputs/opened_img.png', segmented_image_mahalanobis_filtered)
            
            contours_opened, hierarchy = cv.findContours(segmented_image_mahalanobis_filtered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # compare_original_and_segmented_image(img, segmented_image_mahalanobis_filtered, "Filtered Segmented Pumpkins Mahalanobis")


            # 3.2.3 / 3.4.3 - Count orange blobs in image / orthomosaic tile
            # print(f"Pumpkins found in {filename}: {len(contours_opened)}")

            # 3.2.4 - Draw contours
            #contours_img = cv.drawContours(small_img_rgb, contours, -1, (0,0,255), 1)  # Draw all contours
            near_edge = False
            x0 = 0
            y0 = 0
            x1 = 512
            y1 = 512
            temp_cum_sum = 0
            temp_cum_sum_edge = 0
            temp_cum_sum_too_small = 0
            for c in contours_opened:
                M = cv.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    (x, y), radius = cv.minEnclosingCircle(c)
                    radius = int(radius)
                    area = cv.contourArea(c)
                    cum_sum_pump_area.append(area)
                    
                    mean = None
                    if cum_sum_pump_area:
                        mean = np.mean(cum_sum_pump_area)
                    else:
                        mean = area
                    
                    ratio = area/mean
                    
                    large_cluster = False
                    too_small = False
                    # print(f'Area: {area}\nMean: {mean}\nRatio: {ratio}')
                    if ratio > 2:
                        large_cluster = True
                        temp_cum_sum += int(np.round(ratio))
                    elif ratio < 0.15:
                        too_small = True
                        temp_cum_sum += 1
                        temp_cum_sum_too_small += 1
                    else:
                        large_cluster = False
                        temp_cum_sum += 1
                    ratio = int(np.round(ratio))
                    # 3.4.4 - Deal with pumpkins in the overlap, so they are only counted once.
                    if is_circle_near_edge(cx, cy, int(round(radius/2)), x0, y0, x1, y1):
                        near_edge = True
                        # color = (255, 0, 0, 255) # Red
                        color = (0, 0, 255, 255) # Blue
                    else:
                        near_edge = False
                        # color = (0, 0, 255, 255) # Blue
                        color = (0, 255, 0, 255) # Green
                    
                    if large_cluster:
                        color = (255, 0, 255, 255) # Purple
                        # draw extra circles to visualize that the incorrect clusters have been found
                        num_circles = max(1, ratio)
                        for i in range(num_circles):
                            rad = max(5, int(np.round(radius * (1 - 0.3 * i))))  # Gradually decrease size, min 5px
                            cv.circle(img, (cx, cy), rad, color, thickness=2)
                    elif too_small:
                        # print('Found too small')
                        color = (255, 0, 0, 255) # Red
                        cv.circle(img, (cx, cy), radius, color, thickness=2)
                    else:
                        cv.circle(img, (cx, cy), radius, color, thickness=2)
                    
                    if near_edge and large_cluster:
                        temp_cum_sum_edge += ratio
                    elif near_edge and not large_cluster:
                        temp_cum_sum_edge += 1
            
            cum_sum_pump += temp_cum_sum
            cum_sum_edge += temp_cum_sum_edge
            cum_sum_too_small += temp_cum_sum_too_small
            total_pump = int(np.round(cum_sum_pump - cum_sum_too_small - cum_sum_edge/2))
            # if (num_tiles + num_empty) % 16 == 0:
            #     print(f'{filename}: Found {num2str(temp_cum_sum)} pumpkins with {num2str(temp_cum_sum_edge)} near an edge.')
            # plt.imsave('mini_project_1/outputs/input_img_with_counts_edge.png', img)
            # cv.imshow('Contours', cv.cvtColor(img, cv.COLOR_RGB2BGR))
            # cv.waitKey(0)
            # cv.destroyAllWindows()

        # 3.4.5 - Determine amount of pumpkins in the entire field.
        print(f'\nProcessed {num_tiles} tiles, skipped {num_empty} ({percent(num_empty,num_tiles+num_empty)}%) empty tiles.')
        print(f'Found {cum_sum_pump} pumpkins in total.')
        print(f'Of which, {cum_sum_edge} ({percent(cum_sum_edge, cum_sum_pump)}%) were near the edge of an image.')
        print(f'And {cum_sum_too_small} ({percent(cum_sum_too_small, cum_sum_pump)}%) were not actual pumpkins.')
        print(f'With corrections, there was {total_pump} total pumpkins.')
