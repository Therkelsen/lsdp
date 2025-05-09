from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from mpl_toolkits.mplot3d import Axes3D
import csv
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time as tm
import utm

import sys
sys.path.append("mini_project_2/Data for miniproject on visual odometry/2024")

from Map import Map  # type: ignore
from TrackedCamera import TrackedCamera  # type: ignore
from TrackedPoint import TrackedPoint  # type: ignore


def create_output_dir(path):
    # If the folder exists, delete it
    if os.path.exists(path):
        shutil.rmtree(path)

    # Re-create the folder
    os.makedirs(path)


def load_single_sequence(folder_name, sequences_path):
    sequence = []
    all_logs = []
    folder_path = os.path.join(sequences_path, folder_name)
    
    csv_path = os.path.join(folder_path, "frames.csv")
    if not os.path.exists(csv_path):
        print(f"[Warning] Missing CSV in: {folder_path}")
        return sequence, all_logs

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            time = row["time"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            alt = float(row["altitude"])
            sequence_id = int(row["sequence_id"])

            filepath = os.path.join(folder_path, filename)
            frame = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue  # silently skip missing/corrupt images

            global_idx = int(filename.split("_")[1].split(".")[0])
            sequence.append((global_idx, frame, time, (lat, lon, alt), sequence_id))
            all_logs.append(row)

    return sequence, all_logs


def load_sequences(sequences_path):
    all_sequences = []
    all_logs = []

    with ThreadPoolExecutor() as executor:
        future_to_folder = {
            executor.submit(load_single_sequence, folder, sequences_path): folder
            for folder in sorted(os.listdir(sequences_path))
            if os.path.isdir(os.path.join(sequences_path, folder))
        }

        for future in as_completed(future_to_folder):
            sequence, logs = future.result()
            all_sequences.append(sequence)
            all_logs.extend(logs)

    return all_sequences, all_logs


def preprocess_frame(i, frame, K, dist_coeffs, feature_detector, num_imgs):
    """
    Loads, undistorts, and extracts features from the i-th frame.
    """
    if i == 0:
        print("Processing frame")
    if i % max(1, num_imgs // 10) == 0 or i == num_imgs:
        print(f"{i+1}/{num_imgs}...")
    
    # x_corrected = K * (x_distorted - d(x_distorted)), where d(x_distorted) represents the distortion.
    img_undist = cv2.undistort(frame, K, dist_coeffs)
    kp, des = feature_detector.detectAndCompute(img_undist, None)
    return frame, img_undist, kp, des


def calculate_epipolar_distance(pt1, pt2, E):
    """
    Calculate the epipolar distance between two points given the essential matrix.
    
    The epipolar distance is computed by first finding the epipolar line from pt1 using the essential matrix 
    (E) and then calculating the perpendicular distance from pt2 to this line, using the formula:
        distance = |ax + by + c| / sqrt(a^2 + b^2), where (a, b, c) are the coefficients of the epipolar line.
    """
    pt1_homogeneous = np.array([pt1[0], pt1[1], 1.0]).reshape(-1, 1)
    epipolar_line = E.dot(pt1_homogeneous)
    a, b, c = epipolar_line.flatten()
    return abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a**2 + b**2)


def match_and_extract_points(descriptors1, descriptors2, keypoints1, keypoints2, ratio=0.75):
    """
    Match descriptors between two frames using FLANN, apply Lowe's ratio test, and extract matched keypoints.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return good_matches, pts1, pts2


def convert_to_absolute_transformations(rotations, translations):
    """
    Converts relative rotations and translations into absolute transformations.

    For each step i:
        R_abs[i] = R_abs[i-1] dot R_rel[i]
        t_abs[i] = R_abs[i]   dot t_rel[i] + t_abs[i-1]

    Where:
        - @ indicates matrix multiplication
        - R_abs[i]: Absolute rotation at step i
        - t_abs[i]: Absolute translation at step i
        - R_rel[i]: Relative rotation from i-1 to i
        - t_rel[i]: Relative translation from i-1 to i
    """
    filtered_R = [R for R in rotations if R is not None]
    filtered_t = [t.flatten() for t in translations if t is not None]
    
    absolute_rotations = []
    absolute_translations = []
    
    # Initialize the first frame with identity rotation and zero translation
    R_abs = np.eye(3)  # Identity matrix for rotation
    t_abs = np.zeros(3)  # Zero vector for translation
    
    absolute_rotations.append(R_abs)
    absolute_translations.append(t_abs)
    
    for i in range(1, len(filtered_R)):
        # Update the absolute rotation by multiplying the previous absolute rotation with the current relative rotation
        R_abs = np.dot(R_abs, filtered_R[i])
        
        # Update the absolute translation by applying the relative translation, rotated by the previous absolute rotation
        t_abs = np.dot(R_abs, filtered_t[i]) + t_abs
        
        # Append the updated absolute rotation and translation
        absolute_rotations.append(R_abs)
        absolute_translations.append(t_abs)
    
    return absolute_rotations, absolute_translations


def extract_columns_from_folder(folder_path, required_columns):
    """
    Extracts specified columns from all CSV files in the given folder.
    
    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        required_columns (list of str): List of column names to extract.

    Returns:
        list: A list of extracted row data. Each element is a tuple if multiple columns, or a single value if one.
    """
    extracted_data = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file)
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)

                if not header or not all(col in header for col in required_columns):
                    continue

                indices = [header.index(col) for col in required_columns]

                for row in reader:
                    try:
                        values = [parse_value(row[i]) for i in indices]
                        # Unpack single-column result
                        extracted_data.append(values[0] if len(values) == 1 else tuple(values))
                    except (ValueError, IndexError):
                        continue

    return extracted_data

def parse_value(val):
    """Try to convert to int, then float, otherwise keep as string."""
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val.strip()


def visualize_camera_trajectory(sequence_ids, translation_vectors=None, lat_lon_alt=None, points_3d=None,
                                 show=False, save=False, tight=True):
    """
    Visualize the camera trajectory over time, coloring points by sequence ID and lines by time (gradient).
    The camera trajectory and GPS coordinates will be plotted in two separate figures.
    """
    if (not translation_vectors and not lat_lon_alt) or not sequence_ids:
        print("Empty input.")
        return

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    flat_ids = sequence_ids

    # Build consistent color mapping across both plots
    unique_ids = sorted(set(flat_ids))
    seqid_to_color = {sid: colors[i % len(colors)] for i, sid in enumerate(unique_ids)}
    point_colors_all = [seqid_to_color[sid] for sid in flat_ids]

    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    if translation_vectors:
        valid_indices = [i for i, t in enumerate(translation_vectors)
                         if t is not None and t.shape[0] == 3]
        camera_positions = np.array([translation_vectors[i].flatten() for i in valid_indices])
        camera_seq_ids = [flat_ids[i] for i in valid_indices]
        point_colors = [seqid_to_color[sid] for sid in camera_seq_ids]

        scatter = ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                              c=point_colors, label="Trajectory Points", s=20)

        from matplotlib import cm
        times = np.linspace(0, 1, len(camera_positions))
        cmap = cm.get_cmap('viridis')
        for i in range(1, len(camera_positions)):
            seg = np.stack([camera_positions[i-1], camera_positions[i]])
            ax1.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                     color=cmap(times[i]), alpha=0.7, linewidth=2)

        legend_handles = [mpatches.Patch(color=seqid_to_color[sid], label=f"Sequence {sid}")
                          for sid in unique_ids]
        ax1.legend(handles=legend_handles, loc='best')
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.view_init(elev=90, azim=-90)
        ax1.set_title("Camera Trajectories (Translations)")

        max_range = np.ptp(camera_positions, axis=0).max()
        ax1.set_xlim([camera_positions[:, 0].min(), camera_positions[:, 0].min() + max_range])
        ax1.set_ylim([camera_positions[:, 1].min(), camera_positions[:, 1].min() + max_range])
        ax1.set_zlim([camera_positions[:, 2].min(), camera_positions[:, 2].min() + max_range])
    elif points_3d:
        points_3d = np.array(points_3d)

        for i in range(1, len(points_3d)):
            ax1.plot([points_3d[i-1, 0], points_3d[i, 0]],
                     [points_3d[i-1, 1], points_3d[i, 1]],
                     [points_3d[i-1, 2], points_3d[i, 2]],
                     c=plt.cm.viridis(i / len(points_3d)), alpha=0.6)

        # Use same color map as sequence_ids
        points_3d_colors = point_colors_all[:len(points_3d)]
        ax1.scatter(points_3d[:len(points_3d_colors), 0],
                    points_3d[:len(points_3d_colors), 1],
                    points_3d[:len(points_3d_colors), 2],
                    c=points_3d_colors, marker='o', label="X/Y/Z Map Points", s=40)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=90, azim=-90)
        ax1.set_title("Map (X/Y/Z)")
        ax1.legend()

    if lat_lon_alt:
        lat_lon_alt = np.array(lat_lon_alt)
        utm_coords = lat_lon_alt.copy()
        for i, (lat, lon, _) in enumerate(utm_coords):
            x, y, _, _ = utm.from_latlon(lat, lon)
            utm_coords[i, 0] = x
            utm_coords[i, 1] = y

        for i in range(1, len(utm_coords)):
            ax2.plot([utm_coords[i-1, 0], utm_coords[i, 0]],
                     [utm_coords[i-1, 1], utm_coords[i, 1]],
                     [utm_coords[i-1, 2], utm_coords[i, 2]],
                     c=plt.cm.viridis(i / len(utm_coords)), alpha=0.6)

        # Use same color map as sequence_ids
        gps_point_colors = point_colors_all[:len(utm_coords)]
        ax2.scatter(utm_coords[:len(gps_point_colors), 0],
                    utm_coords[:len(gps_point_colors), 1],
                    utm_coords[:len(gps_point_colors), 2],
                    c=gps_point_colors, marker='o', label="UTM X/UTM Y/Alt Points", s=40)

        ax2.set_xlabel("UTM X")
        ax2.set_ylabel("UTM Y")
        ax2.set_zlabel("Altitude (m)")
        ax2.view_init(elev=90, azim=-90)
        ax2.set_title("UTM (X/Y/Alt)")
        ax2.legend()

    if tight:
        plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight')
        print(f"Saved trajectory to {save}")

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
    - cut_outliers: If set (e.g. 3), cuts data outside of mean \u00B1 cut_outliers * \u03C3.
    - show: If True, shows the plot.
    - save: If True, saves the plot.
    - tight: If True, applies tight layout.
    """
    data = np.asarray(data)
    original_len = len(data)

    if cut_outliers:
        mean = np.mean(data)
        std = np.std(data)
        cutoff = cut_outliers * std
        data = data[np.abs(data - mean) <= cutoff]

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
    plt.title(title_and_xlabel)
    plt.xlabel(title_and_xlabel)
    plt.ylabel(ylabel)

    if tight:
        plt.tight_layout()

    if save:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Histogram saved to {filename}")

    if show:
        plt.show()

    print(f"Histogram plotted for '{title_and_xlabel}': {len(data)} values (filtered from {original_len})")


if __name__ == "__main__":
    start_time = tm.time()

    sequences_path = "mini_project_2/saved_sequences"
    
    output_path = "mini_project_2/output/"
    kp_path = os.path.join(output_path, "keypoints/")
    match_path = os.path.join(output_path, "matches/")
    # create_output_dir(kp_path)
    # create_output_dir(match_path)
    
    all_sequences, all_logs = load_sequences(sequences_path)

    print(f"Loaded {len(all_sequences)} sequences.")
    flattened_sequences = [item for seq in all_sequences for item in seq]

    # Camera intrinsic parameters (from calibration)
    f = 2676.1051390718389      # Focal length in pixels
    cx = -35.243952918157035    # Principal point X-coordinate (in pixels)
    cy = -279.58562078697361    # Principal point Y-coordinate (in pixels)
    k1 = 0.0097935857180804498  # Radial distortion coefficient k1
    k2 = -0.021794052829051412  # Radial distortion coefficient k2
    k3 = 0.017776502734846815   # Radial distortion coefficient k3
    p1 = 0.0046443590741258711  # Tangential distortion coefficient p1
    p2 = -0.0045664024579022498 # Tangential distortion coefficient p2
    
    # Camera intrinsic matrix
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    # Distortion coefficients array (order: k1, k2, p1, p2, k3)
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    feature_detector = cv2.SIFT_create()

    num_imgs = len(flattened_sequences)
    print(f"Total number of frames to process: {num_imgs}")

    if num_imgs < 2:
        raise ValueError("Not enough frames for stereo matching. You need at least two frames.")

    all_imgs = [None] * num_imgs
    all_imgs_undist = [None] * num_imgs
    all_keypoints = [None] * num_imgs
    all_descriptors = [None] * num_imgs

    all_good_matches = []
    all_pts_pairs = [None] * num_imgs
    all_essential_matrices = [None] * num_imgs
    all_masks = [None] * num_imgs
    all_distances = []
    all_img_kp_pairs = [None] * num_imgs
    all_img_matches = []
    all_R_matrices_relative = [None] * num_imgs
    all_t_vectors_relative = [None] * num_imgs
    all_R_matrices_absolute = [None] * num_imgs
    all_t_vectors_absolute = [None] * num_imgs

    map = Map()
    map.camera_matrix = K
    all_cams = []
    all_proj_mats = []

    print(f"Running vSLAM algorithm on {num_imgs} images...")
    ############### Exercise 9.4.1 ###############
    for i, (global_idx, frame, time, attitude, sequence) in enumerate(flattened_sequences[:num_imgs]):
        if frame is not None:
            ############### Exercise 9.2.1 ###############
            # We chose SIFT
            all_imgs[i], all_imgs_undist[i], all_keypoints[i], all_descriptors[i] = preprocess_frame(
                i, frame, K, dist_coeffs, feature_detector, num_imgs
            )
        else:
            continue

        if i == 0:
            # For the first frame, initialize the absolute pose:
            R0 = np.eye(3)                  # Identity rotation (no rotation)
            t0 = np.zeros((3, 1))           # Zero translation (origin)
            all_R_matrices_absolute = R0    # Store initial absolute rotation
            all_t_vectors_absolute = t0     # Store initial absolute translation
            
            # Create and add the first TrackedCamera to the map
            cam0 = TrackedCamera(R=R0, t=t0, frame_id=0, frame=all_imgs_undist[0], camera_id=None)
            cam0 = map.add_camera(cam0)
            all_cams.append(cam0)
            
            # Store the initial projection matrix for triangulation
            all_proj_mats.append(K @ np.hstack((R0, t0)))
            continue

        good_matches, pts1, pts2 = match_and_extract_points(
            all_descriptors[i - 1], all_descriptors[i],
            all_keypoints[i - 1], all_keypoints[i]
        )
        all_good_matches.append(good_matches)
        all_pts_pairs[i - 1] = (pts1, pts2)

        if len(good_matches) < 5:
            continue
        
        ############### Exercise 9.2.2 ###############
        # We are using the OpenCV method findEssentialMat for this.
        # Estimate the essential matrix between matched points using RANSAC for robust outlier rejection
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        all_essential_matrices[i - 1] = E
        all_masks[i - 1] = mask
        
        ############### Exercise 9.2.3 ###############
        # Calculate epipolar distances for all good
        # matches using the estimated essential matrix
        distances = [calculate_epipolar_distance(all_keypoints[i - 1][m.queryIdx].pt, 
                                                 all_keypoints[i][m.trainIdx].pt, E) for m in good_matches]
        all_distances.append(distances)

        # Draw keypoints on the undistorted images for visualization
        img1_kp = cv2.drawKeypoints(all_imgs_undist[i - 1], all_keypoints[i - 1], None, color=(0, 255, 0))
        img2_kp = cv2.drawKeypoints(all_imgs_undist[i], all_keypoints[i], None, color=(0, 255, 0))
        
        
        
        # cv2.imwrite(f"{kp_path}frame{i-1}_keypoints.png", img1_kp)
        # cv2.imwrite(f"{kp_path}frame{i}_keypoints.png", img2_kp)
        
        all_img_kp_pairs[i - 1] = (img1_kp, img2_kp)

        # Draw matches between the two frames for visualization
        img_matches = cv2.drawMatches(
            all_imgs_undist[i - 1], all_keypoints[i - 1],
            all_imgs_undist[i], all_keypoints[i],
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # cv2.imwrite(f"{match_path}matches_btwn_frame{i-1}_and_{i}.png", img_matches)

        ############### Exercise 9.2.4 ###############
        # Recover the relative camera rotation and translation
        # from the essential matrix and matched points
        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1, pts2, K)

        all_R_matrices_relative[i - 1] = R_rel
        all_t_vectors_relative[i - 1] = t_rel

        
        ############### Exercise 9.3.2 ###############
        img = all_imgs_undist[i]
        cam = TrackedCamera(R=R_rel, t=t_rel, frame_id=i, frame=img, camera_id=None)
        cam = map.add_camera(cam)
        all_cams.append(cam)

        # Append the projection matrix for the current camera (used for triangulation)
        all_proj_mats.append(K @ np.hstack((R_rel, t_rel)))

        pts_flat = [pts.reshape(-1, 2).T for pts in (pts1, pts2)]
        # Triangulate 3D points from the matched keypoints using
        # the projection matrices of the two views
        
        # Homogeneous coordinates
        points_4d = cv2.triangulatePoints(all_proj_mats[i - 1], all_proj_mats[i], *pts_flat)
        # Transform to 3D
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

            for cam_obj, pts in zip([all_cams[-2], all_cams[-1]], [pts1, pts2]):
                obs = type('Observation', (), {})()
                obs.camera_id = cam_obj.camera_id
                obs.point_id = tracked_point.point_id
                obs.image_coordinates = pts[j, 0]
                map.observations.append(obs)

    ############### Exercise 9.3.3 ###############
    reprojection_errors, total_reprojection_error = map.calculate_reprojection_error()
    print(f"\nTotal Reprojection Error: {total_reprojection_error}")

    plot_histogram(reprojection_errors,
                title_and_xlabel="Reprojection Error Distribution (Before Bundle Adjustment)",
                ylabel="Density", filename="", cut_outliers=False, show=False, save=False, tight=True)

    ############### Exercise 9.3.4 ###############
    map.optimize_map()

    reprojection_errors, total_reprojection_error = map.calculate_reprojection_error()
    print(f"\nTotal Reprojection Error: {total_reprojection_error}")

    plot_histogram(reprojection_errors,
                title_and_xlabel="Reprojection Error Distribution (After Bundle Adjustment)",
                ylabel="Density", filename="", cut_outliers=False, show=False, save=False, tight=True)

    std_error = np.std(reprojection_errors)
    map.remove_observations_with_reprojection_errors_above_threshold(3 * std_error)
    map.optimize_map()

    reprojection_errors, total_reprojection_error = map.calculate_reprojection_error()
    print(f"\nTotal Reprojection Error (After Cutting μ \u00B1 3\u03C3 Observations): {total_reprojection_error}")

    plot_histogram(reprojection_errors,
                title_and_xlabel="Reprojection Error Distribution (After Cutting μ \u00B1 3\u03C3 Observations)",
                ylabel="Density", filename="", cut_outliers=False, show=False, save=False, tight=True)
    
    ############### Exercise 9.4.4 ###############
    # Convert relative rotations and translations to absolute camera poses
    absolute_R, absolute_t = convert_to_absolute_transformations(all_R_matrices_relative, all_t_vectors_relative)
    
    # points_array = [tp.point for tp in map.points]

    # # Ensure absolute_t has the same length as the number of frames/IDs
    # if len(absolute_t) < len(flattened_sequences):
    #     absolute_t = [np.zeros(3)] + absolute_t

    print(f"\nDone processing {num_imgs} images and triangulating points.")

    all_sequence_ids = extract_columns_from_folder("mini_project_2/saved_sequences/", ["sequence_id"])
    all_gps_coordinates = extract_columns_from_folder("mini_project_2/saved_sequences/", ["latitude", "longitude", "altitude"])

    visualize_camera_trajectory(sequence_ids=all_sequence_ids[:num_imgs], translation_vectors=absolute_t[:num_imgs],
                                lat_lon_alt=all_gps_coordinates[:num_imgs], show=False)

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

    print(f"\nMap Statistics:")
    print(f"Number of Images: {len(map.cameras)}")
    print(f"Number of Points: {len(map.points)}")
    print(f"Number of Observations: {len(reprojection_errors)}")

    end_time = tm.time()

    elapsed_time = end_time - start_time

    print(f"Execution time: {elapsed_time:.4f} seconds")

    plt.show()
