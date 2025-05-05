from mpl_toolkits.mplot3d import Axes3D
import chardet
import csv
import cv2
import g2o # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append("mini_project_2/Data for miniproject on visual odometry/2024")

from Map import Map
from TrackedCamera import TrackedCamera
from TrackedPoint import TrackedPoint


def extract_sequences_from_log(csv_path):
    # Load the entire CSV as a structured array, assuming header is present
    data = np.genfromtxt(csv_path, delimiter=',', dtype=None, names=True, encoding='utf-8')

    if 'CUSTOMisVideo' not in data.dtype.names:
        raise ValueError(f"'CUSTOMisVideo' column not found in the CSV file.")
    
    is_video = data['CUSTOMisVideo']
    
    sequences = []
    current_sequence = []

    for i, status in enumerate(is_video):
        # If the status is "Recording", add the index to the current sequence
        if status == "Recording":
            current_sequence.append(i)

        # If the status is "Stop" or we are at the last row and still recording, finalize the sequence
        if status == "Stop" or (i == len(is_video) - 1 and status == "Recording"):
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
    return sequences


def preprocess_frame(i, frame_files, frames_path, K, dist_coeffs, feature_detector, num_imgs):
    """
    Loads, undistorts, and extracts features from the i-th frame.
    Updates all_imgs, all_imgs_undist, all_keypoints, and all_descriptors in place.
    Parameters:
    - i: Index of the frame to process.
    - frame_files: List of frame filenames.
    - frames_path: Path to the directory containing the frames.
    - K: Intrinsic camera matrix.
    - dist_coeffs: Distortion coefficients.
    - feature_detector: Feature detector object (e.g., SIFT).
    - num_imgs: Total number of images.
    """
    print(f"\nPre-processing frame {i+1} out of {num_imgs}...")
    # Load the image
    print(f"Loading image {i}...")
    img = cv2.imread(os.path.join(frames_path, frame_files[i]), cv2.IMREAD_GRAYSCALE)

    # Undistort the image
    print(f"Undistorting image {i}...")
    img_undist = cv2.undistort(img, K, dist_coeffs)

    # Detect all_keypoints and compute all_descriptors
    print(f"Detecting all_keypoints and computing all_descriptors for image {i}...")
    kp, des = feature_detector.detectAndCompute(img_undist, None)
    
    return img, img_undist, kp, des


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


def match_and_extract_points(descriptors1, descriptors2, keypoints1, keypoints2, bf_matcher, ratio=0.75):
    """
    Match descriptors between two frames, apply Lowe's ratio test, and extract matched keypoints.

    Parameters:
    - descriptors1: Descriptors from the first image.
    - descriptors2: Descriptors from the second image.
    - keypoints1: Keypoints from the first image.
    - keypoints2: Keypoints from the second image.
    - bf_matcher: OpenCV BFMatcher object.
    - ratio: Lowe's ratio threshold.

    Returns:
    - matches: All matches (list of DMatch pairs).
    - good_matches: Matches passing Lowe's ratio test.
    - pts1: Matched points from the first image (Nx1x2).
    - pts2: Matched points from the second image (Nx1x2).
    """
    print("Matching descriptors...")
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return matches, good_matches, pts1, pts2


def convert_to_absolute_transformations(rotations, translations):
    """
    Converts relative rotations and translations into absolute transformations.
    
    Parameters:
    - rotations: List of 3x3 rotation matrices (relative rotations).
    - translations: List of 3x1 translation vectors (relative translations).
    
    Returns:
    - absolute_rotations: List of 3x3 rotation matrices (absolute rotations).
    - absolute_translations: List of 3x1 translation vectors (absolute translations).
    """
    
    filtered_R_matrices = [R for R in rotations if R is not None]
    filtered_t_vectors = [t.flatten() for t in translations if t is not None]
    
    # Initialize with the first frame's rotation (identity) and translation (zero vector)
    absolute_rotations = [np.eye(3)]  # Identity rotation for the first frame
    absolute_translations = [np.zeros(3)]  # Zero translation for the first frame

    # Iterate through the rest of the frames
    for i in range(1, len(filtered_R_matrices)):
        # Get the previous absolute rotation and translation
        prev_rotation = absolute_rotations[-1]
        prev_translation = absolute_translations[-1]
        
        # Relative rotation and translation for the current frame
        relative_rotation = filtered_R_matrices[i]
        relative_translation = filtered_t_vectors[i]
        
        # Calculate the absolute rotation by applying the relative rotation
        absolute_rotation = np.dot(prev_rotation, relative_rotation)
        
        # Calculate the absolute translation by applying the relative translation
        absolute_translation = np.dot(prev_rotation, relative_translation) + prev_translation
        
        # Append to the result lists
        absolute_rotations.append(absolute_rotation)
        absolute_translations.append(absolute_translation)

    return absolute_rotations, absolute_translations


def visualize_camera_trajectory(all_t_vectors, log_file_path=None, show=False, save=False, tight=True):
    """
    Visualize the camera trajectory over time and compare it with the trajectory from a log file.
    """
    # Filter out None values and invalid translation vectors (ensure they are 3-dimensional)
    valid_t_vectors = [t for t in all_t_vectors if t is not None and t.shape == (3,)]
    if not valid_t_vectors:
        print("No valid translation vectors to plot.")
        return

    # Extract the camera positions from translation vectors
    camera_positions = np.array([t.flatten() for t in valid_t_vectors])
    
    # Create an array of indices or time steps to use for color mapping
    times = np.linspace(0, len(camera_positions) - 1, len(camera_positions))

    # Plot the camera trajectory with a color gradient based on time
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the line connecting the points
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='b', label="Estimated Trajectory", linestyle='-', alpha=0.6)

    # Scatter plot for the colored points
    img = ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c=times, cmap='viridis', label="Trajectory Points")
    fig.colorbar(img, ax=ax, label='Time/Index')  # Add color bar for reference

    if log_file_path:
        # Load the true camera trajectory from the log file (assuming log file contains x, y, z columns)
        true_trajectory = np.loadtxt(log_file_path, delimiter=',')
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], label="True Trajectory", color="r", linestyle='--')

    # Labeling the plot
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    plt.title("Camera Trajectory")
    plt.legend()

    # Get the min and max values for each axis
    x_min, x_max = camera_positions[:, 0].min(), camera_positions[:, 0].max()
    y_min, y_max = camera_positions[:, 1].min(), camera_positions[:, 1].max()
    z_min, z_max = camera_positions[:, 2].min(), camera_positions[:, 2].max()

    min_range = min(x_min, y_min, z_min)
    max_range = max(x_max, y_max, z_max)
    
    # Set the limits for each axis
    # ax.set_xlim([min_range, max_range])
    # ax.set_ylim([min_range, max_range])
    # ax.set_zlim([min_range, max_range])

    # Set equal scaling for all axes based on the maximum range (1:1:1 ratio)
    ax.set_box_aspect([1, 1, 1])  # Equal scaling along x, y, and z axes

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
        print(f"Saved camera trajectory plot to {save}")
    if show:
        plt.show()



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

log_file_path = 'mini_project_2/Data for miniproject on visual odometry/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv'

log_data = None
with open(log_file_path, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    log_data = list(reader)
    
sequences = extract_sequences_from_log(log_file_path)
print(f"Extracted {len(sequences)} sequences from the log file.")
# Get the first sequence
if sequences:
    for i, seq in enumerate(sequences):
        print(f"Sequence {i}: {seq}")

exit()

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

all_imgs = [None] * num_imgs
all_imgs_undist = [None] * num_imgs
all_keypoints = [None] * num_imgs
all_descriptors = [None] * num_imgs

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
all_matches = []
all_good_matches = []
all_pts_pairs = [None] * num_imgs
all_essential_matrices = [None] * num_imgs
all_masks = [None] * num_imgs
all_distances = []
all_img_kp_pairs = [None] * num_imgs
all_img_matches = []
all_R_matrices = [None] * num_imgs
all_t_vectors = [None] * num_imgs

map = Map()
map.camera_matrix = K
all_cams = []
all_proj_mats = []

print(f"Running vSLAM algorithm on {num_imgs} images...")
for i in range(num_imgs):
    print("\n===========================")
    all_imgs[i], all_imgs_undist[i], all_keypoints[i], all_descriptors[i] = preprocess_frame(i, frame_files, frames_path, K, dist_coeffs, feature_detector, num_imgs)
    
    if i == 0:
        # Add first camera at identity pose
        R0 = np.eye(3)
        t0 = np.zeros((3, 1))
        cam0 = TrackedCamera(R=R0, t=t0, frame_id=0, frame=all_imgs_undist[0], camera_id=None)
        cam0 = map.add_camera(cam0)  # Get assigned camera_id
        all_cams.append(cam0)
        all_proj_mats.append(K @ np.hstack((R0, t0)))
        # Two images are needed for everything following
        continue
    if i == num_imgs:
        # Stupid indexing
        break
    
    print(f"\nProcessing frame pair {i} out of {num_imgs - 1} total pairs ({num_imgs} images)...")

    # Match all_descriptors between previous and current frame
    matches, good_matches, pts1, pts2 = match_and_extract_points(
        all_descriptors[i - 1], all_descriptors[i],
        all_keypoints[i - 1], all_keypoints[i],
        bf
    )
    all_matches.append(matches)
    all_good_matches.append(good_matches)
    all_pts_pairs[i - 1] = (pts1, pts2)

    if len(good_matches) < 5:
        print(f"Not enough good matches between frames {i-1} and {i}, skipping.")
        continue

    print(f"Estimating essential matrix between frames {i-1} and {i}...")
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    all_essential_matrices[i - 1] = E
    all_masks[i - 1] = mask

    # Calculate epipolar distance for each matched point
    distances = []
    for m in good_matches:
        pt1 = all_keypoints[i - 1][m.queryIdx].pt
        pt2 = all_keypoints[i][m.trainIdx].pt
        distance = calculate_epipolar_distance(pt1, pt2, E)
        distances.append(distance)
    all_distances.append(distances)

    img1_kp = cv2.drawKeypoints(all_imgs_undist[i - 1], all_keypoints[i - 1], None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(all_imgs_undist[i], all_keypoints[i], None, color=(0, 255, 0))
    all_img_kp_pairs[i - 1] = (img1_kp, img2_kp)

    img_matches = cv2.drawMatches(
        all_imgs_undist[i - 1], all_keypoints[i - 1],
        all_imgs_undist[i], all_keypoints[i],
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    all_R_matrices[i - 1] = R
    all_t_vectors[i - 1] = t

    print("\nRelative Motion:")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")

    img = all_imgs_undist[i]
    R = all_R_matrices[i - 1]
    t = all_t_vectors[i - 1]
    
    cam = TrackedCamera(R=R, t=t, frame_id=i, frame=img, camera_id=None)
    cam = map.add_camera(cam)  # Get assigned camera_id
    all_cams.append(cam)
    
    all_proj_mats.append(K @ np.hstack((R, t)))
    
    good_matches = all_good_matches[i - 1]
    pts1, pts2 = all_pts_pairs[i - 1]
    if pts1.shape[0] < 5:
        print(f"Not enough points for triangulation between frames {i-1} and {i}, skipping.")
        continue
    
    pts_flat = [pts.reshape(-1, 2).T for pts in (pts1, pts2)]
    points_4d = cv2.triangulatePoints(all_proj_mats[i - 1], all_proj_mats[i], *pts_flat)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    for j, pt3d in enumerate(points_3d):
        tracked_point = TrackedPoint(
            point=pt3d,
            descriptor=all_descriptors[i - 1][good_matches[j].queryIdx],
            color=None,
            feature_id=good_matches[j].queryIdx,
            point_id=None
        )
        tracked_point = map.add_point(tracked_point)
        
        # Use the correct camera_id from all_cams[-2] and all_cams[-1]
        for cam_obj, pts in zip([all_cams[-2], all_cams[-1]], [pts1, pts2]):
            obs = type('Observation', (), {})()
            obs.camera_id = cam_obj.camera_id  # Use assigned camera_id
            obs.point_id = tracked_point.point_id
            obs.image_coordinates = pts[j, 0]
            map.observations.append(obs)
        
        # print(f"3D Point {tracked_point.point_id}")


absolute_R, absolute_t = convert_to_absolute_transformations(all_R_matrices, all_t_vectors)

print(f"\nDone processing {num_imgs} images and triangulating points.")
# Visualize camera trajectory
visualize_camera_trajectory(absolute_t, log_file_path=None, show=False, save="", tight=True)

# Calculate epipolar distances for all pairs
all_distances_flat = np.concatenate(all_distances) if all_distances else np.array([])

plot_histogram(
    all_distances_flat,
    title_and_xlabel="Epipolar Distance (All Pairs)",
    ylabel="Density",
    filename="",
    cut_outliers=3,
    show=False,
    save=False,
    tight=True
)
        
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