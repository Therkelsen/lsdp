import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import g2o # type: ignore

import sys
sys.path.append("mini_project_2/Data for miniproject on visual odometry/2024")

from Map import Map
from TrackedCamera import TrackedCamera
from TrackedPoint import TrackedPoint


def calculate_epipolar_distance(pt1, pt2, E):
    """
    Calculate the epipolar distance between two points given the essential matrix.

    Parameters:
    - pt1: Point in the first image (x, y).
    - pt2: Point in the second image (x, y).
    - E: Essential matrix.
    Returns:
    - distance: Epipolar distance.
    """
    # Convert points to homogeneous coordinates
    pt1_homogeneous = np.array([pt1[0], pt1[1], 1.0]).reshape(-1, 1)

    # Compute the epipolar line in the second image
    epipolar_line = E.dot(pt1_homogeneous)
    
    # Epipolar line equation: ax + by + c = 0
    a, b, c = epipolar_line.flatten()
    
    # Compute the distance from pt2 to the epipolar line
    distance = abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a**2 + b**2)
    return distance


def plot_histogram(data, title_and_xlabel, ylabel, filename, cut_outliers=False, show=False, save=False, tight=True):
    """
    Plot a histogram of the given data with optional outlier removal.
    
    Parameters:
    - data: The data to plot (array-like).
    - title_and_xlabel: Title and X-axis label.
    - ylabel: Label for the Y-axis.
    - filename: Filename to save the plot if save=True.
    - cut_outliers: If set (e.g. 3), cuts data outside of mean ± cut_outliers * std.
    - show: If True, shows the plot.
    - save: If True, saves the plot.
    - tight: If True, applies tight layout.
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    outlier_title = ""

    if cut_outliers:
        # Remove outliers beyond ±cut_outliers * std
        cutoff = cut_outliers * std
        data = data[np.abs(data - mean) <= cutoff]

        # Recalculate stats
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        outlier_title = f" (Within μ \u00B1 3\u03C3)"

    # Print stats
    print(f"\n{title_and_xlabel}")
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print(f"Max: {max_val:.2f}")
    print(f"Min: {min_val:.2f}")

    # # Freedman-Diaconis rule for bin width
    # iqr = np.percentile(data, 75) - np.percentile(data, 25)
    # bin_width = 2 * iqr * len(data) ** (-1/3) if iqr > 0 else std / 5
    # bin_width = max(bin_width, 1e-6)  # Avoid division by zero
    # bins = max(1, int(np.ceil((data.max() - data.min()) / bin_width)))

    # Plot histogram
    # just let it figure out bins by itself
    plt.figure(figsize=(8, 5))
    # plt.hist(data, bins=bins, density=True, alpha=0.6, color='g')
    plt.hist(data, bins=20, density=True, alpha=0.6, color='g')

    # Vertical reference lines
    plt.axvline(min_val, color='c', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.2f}')
    plt.axvline(mean - std, color='b', linestyle='dashed', linewidth=2, label=f'Mean - \u03C3: {mean - std:.2f}')
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std, color='b', linestyle='dashed', linewidth=2, label=f'Mean + \u03C3: {mean + std:.2f}')
    plt.axvline(max_val, color='y', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.2f}')

    plt.xlabel(title_and_xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Histogram of {title_and_xlabel}{outlier_title}")
    plt.legend()

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved histogram to {filename}")
    if show:
        plt.show()


################# Exercise 9.2 #################

# Path to saved frames (make sure these are ordered correctly!)
frames_path = 'mini_project_2/saved_frames'
frame_files = sorted(os.listdir(frames_path))

# Intrinsic parameters from XML
f = 2676.1051390718389  # Focal length
cx = -35.243952918157035  # Principal point x-coordinate
cy = -279.58562078697361  # Principal point y-coordinate
k1 = 0.0097935857180804498  # Radial distortion coefficient 1
k2 = -0.021794052829051412  # Radial distortion coefficient 2
k3 = 0.017776502734846815  # Radial distortion coefficient 3
p1 = 0.0046443590741258711  # Tangential distortion coefficient 1
p2 = -0.0045664024579022498  # Tangential distortion coefficient 2

# Construct the intrinsic camera matrix K
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

# Distortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Create SIFT detector
feature_detector = cv2.SIFT_create()

num_imgs = len(frame_files)
# num_imgs = 6

# Ensure num_imgs is even for stereo matching
num_imgs = num_imgs - (num_imgs % 2)

imgs = []
imgs_undist = []
keypoints = []
descriptors = []

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
all_matches = []
all_good_matches = []
all_pts_pairs = []
all_essential_matrices = []
all_masks = []
all_distances = []
all_img_kp_pairs = []
all_img_matches = []
all_R_matrices = []
all_t_vectors = []

map = Map()
map.camera_matrix = K
cams = []
proj_mats = []

print(f"Running VSLAM algorithm on {num_imgs} images...")
for i in range(num_imgs):
    print("\n===========================")
    print(f"\nPre-processing frame {i+1} out of {num_imgs}...")
    # Load the image
    print(f"Loading image {i}...")
    img = cv2.imread(os.path.join(frames_path, frame_files[i]), cv2.IMREAD_GRAYSCALE)
    imgs.append(img)
    
    # Undistort the image
    print(f"Undistorting image {i}...")
    img_undist = cv2.undistort(img, K, dist_coeffs)
    imgs_undist.append(img_undist)
    
    # Detect keypoints and compute descriptors
    print(f"Detecting keypoints and computing descriptors for image {i}...")
    kp, des = feature_detector.detectAndCompute(img_undist, None)
    keypoints.append(kp)
    descriptors.append(des)
    
    if i == 0:
        # Add first camera at identity pose
        R0 = np.eye(3)
        t0 = np.zeros((3, 1))
        cam0 = TrackedCamera(R=R0, t=t0, frame_id=0, frame=imgs_undist[0], camera_id=None)
        cam0 = map.add_camera(cam0)  # Get assigned camera_id
        cams.append(cam0)
        proj_mats.append(K @ np.hstack((R0, t0)))
        # Two images are needed for everything following
        continue
    if i == num_imgs:
        # Stupid indexing
        break
    
    print(f"\nProcessing frame pair {i} out of {num_imgs - 1} total pairs ({num_imgs} images)...")

    # Match descriptors between previous and current frame
    print(f"Matching descriptors between frames {i-1} and {i}...")
    matches = bf.knnMatch(descriptors[i - 1], descriptors[i], k=2)
    all_matches.append(matches)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    all_good_matches.append(good_matches)

    # Get the points from the good matches
    pts1 = np.float32([keypoints[i - 1][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints[i][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    all_pts_pairs.append((pts1, pts2))

    if len(good_matches) < 5:
        print(f"Not enough good matches between frames {i-1} and {i}, skipping.")
        continue

    print(f"Estimating essential matrix between frames {i-1} and {i}...")
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    all_essential_matrices.append(E)
    all_masks.append(mask)

    # Calculate epipolar distance for each matched point
    distances = []
    for m in good_matches:
        pt1 = keypoints[i - 1][m.queryIdx].pt
        pt2 = keypoints[i][m.trainIdx].pt
        distance = calculate_epipolar_distance(pt1, pt2, E)
        distances.append(distance)
    all_distances.append(distances)

    distances = np.array(distances)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    print(f"Mean Epipolar Distance: {mean_distance}")
    print(f"Standard Deviation of Epipolar Distance: {std_distance}")
    print(f"Max Epipolar Distance: {max_distance}")
    print(f"Min Epipolar Distance: {min_distance}")

    distances_filtered = distances[np.abs(distances - mean_distance) <= 3 * std_distance]

    img1_kp = cv2.drawKeypoints(imgs_undist[i - 1], keypoints[i - 1], None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(imgs_undist[i], keypoints[i], None, color=(0, 255, 0))
    all_img_kp_pairs.append((img1_kp, img2_kp))

    img_matches = cv2.drawMatches(
        imgs_undist[i - 1], keypoints[i - 1],
        imgs_undist[i], keypoints[i],
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("mini_project_2/exercise_9_2_data/frame_matches.png", img_matches)

    plt.hist(distances_filtered, bins=20)
    plt.xlabel('Epipolar Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Epipolar Distances (Filtered)')
    # plt.savefig("mini_project_2/exercise_9_2_data/epipolar_distance_histogram_filtered.png")

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    all_R_matrices.append(R)
    all_t_vectors.append(t)

    print("\nRelative Motion:")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")

    img = imgs_undist[i]
    R = all_R_matrices[i - 1]
    t = all_t_vectors[i - 1]
    
    cam = TrackedCamera(R=R, t=t, frame_id=i, frame=img, camera_id=None)
    cam = map.add_camera(cam)  # Get assigned camera_id
    cams.append(cam)
    
    proj_mats.append(K @ np.hstack((R, t)))
    
    good_matches = all_good_matches[i - 1]
    pts1, pts2 = all_pts_pairs[i - 1]
    if pts1.shape[0] < 5:
        print(f"Not enough points for triangulation between frames {i-1} and {i}, skipping.")
        continue
    
    pts_flat = [pts.reshape(-1, 2).T for pts in (pts1, pts2)]
    points_4d = cv2.triangulatePoints(proj_mats[i - 1], proj_mats[i], *pts_flat)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    for j, pt3d in enumerate(points_3d):
        tracked_point = TrackedPoint(
            point=pt3d,
            descriptor=descriptors[i - 1][good_matches[j].queryIdx],
            color=None,
            feature_id=good_matches[j].queryIdx,
            point_id=None
        )
        tracked_point = map.add_point(tracked_point)
        
        # Use the correct camera_id from cams[-2] and cams[-1]
        for cam_obj, pts in zip([cams[-2], cams[-1]], [pts1, pts2]):
            obs = type('Observation', (), {})()
            obs.camera_id = cam_obj.camera_id  # Use assigned camera_id
            obs.point_id = tracked_point.point_id
            obs.image_coordinates = pts[j, 0]
            map.observations.append(obs)
        
        # print(f"3D Point {tracked_point.point_id}")

print(f"\nDone processing {num_imgs} images and triangulating points.")
        
# Calculate reprojection error
reprojection_errors, total_reprojection_error = map.calculate_reprojection_error()

print("\nReprojection Error Statistics (Before Bundle Adjustment):")
print(f"Total Reprojection Error: {total_reprojection_error}")


plot_histogram(reprojection_errors,
               title_and_xlabel="Reprojection Error Distribution (Before Bundle Adjustment)",
               ylabel="Density", filename="", cut_outliers=False, show=False, save=False, tight=True)

map.optimize_map()

reprojection_errors, total_reprojection_error = map.calculate_reprojection_error()

print("\nReprojection Error Statistics (After Bundle Adjustment):")
print(f"Total Reprojection Error: {total_reprojection_error}")


plot_histogram(reprojection_errors,
               title_and_xlabel="Reprojection Error Distribution (After Bundle Adjustment)",
               ylabel="Density", filename="", cut_outliers=False, show=False, save=False, tight=True)

# Cut observations outside ±3\u03C3 after BA
std_error = np.std(reprojection_errors)
map.remove_observations_with_reprojection_errors_above_threshold(3 * std_error)
map.optimize_map()

reprojection_errors, total_reprojection_error = map.calculate_reprojection_error()

print(f"\nReprojection Error Statistics (After Cutting μ \u00B1 3\u03C3 Observations):")
print(f"Total Reprojection Error: {total_reprojection_error}")

plot_histogram(reprojection_errors,
               title_and_xlabel="Reprojection Error Distribution (After Cutting μ \u00B1 3\u03C3 Observations)",
               ylabel="Density", filename="", cut_outliers=False, show=False, save=False, tight=True)


# now lets print some stats about the map like how many images and cameras and total points etc.
print(f"\nMap Statistics:")
print(f"Number of Images: {len(map.cameras)}")
print(f"Number of Points: {len(map.points)}")
print(f"Number of Observations: {len(reprojection_errors)}")

plt.show()