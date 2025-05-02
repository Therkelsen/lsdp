import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("mini_project_2/Data for miniproject on visual odometry/2024")

from Map import Map
from TrackedCamera import TrackedCamera
from TrackedPoint import TrackedPoint


################# Exercise 9.2 #################

# Path to saved frames (make sure these are ordered correctly!)
frames_path = 'mini_project_2/saved_frames'
frame_files = sorted(os.listdir(frames_path))

# Load the first two frames
img1 = cv2.imread(os.path.join(frames_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(frames_path, frame_files[1]), cv2.IMREAD_GRAYSCALE)

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

# Undistort the images
img1_undistorted = cv2.undistort(img1, K, dist_coeffs)
img2_undistorted = cv2.undistort(img2, K, dist_coeffs)

# Create SIFT detector
feature_detector = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = feature_detector.detectAndCompute(img1_undistorted, None)
keypoints2, descriptors2 = feature_detector.detectAndCompute(img2_undistorted, None)

# Match descriptors using Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Lowe's ratio test threshold
        good_matches.append(m)

# Get the points from the good matches
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate the essential matrix using the camera intrinsic matrix
# Estimate the essential matrix using RANSAC for robustness
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Function to calculate the distance between a point and the epipolar line
def calculate_epipolar_distance(pt1, pt2, E, K):
    # Convert points to homogeneous coordinates
    pt1_homogeneous = np.array([pt1[0], pt1[1], 1.0]).reshape(-1, 1)

    # Compute the epipolar line in the second image
    epipolar_line = E.dot(pt1_homogeneous)
    
    # Epipolar line equation: ax + by + c = 0
    a, b, c = epipolar_line.flatten()
    
    # Compute the distance from pt2 to the epipolar line
    distance = abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a**2 + b**2)
    return distance

# Calculate epipolar distances for all matched points
distances = []
for m in good_matches:
    pt1 = keypoints1[m.queryIdx].pt
    pt2 = keypoints2[m.trainIdx].pt
    distance = calculate_epipolar_distance(pt1, pt2, E, K)
    distances.append(distance)

# Convert distances to numpy array for easier statistical analysis
distances = np.array(distances)

# Calculate summary statistics
mean_distance = np.mean(distances)
std_distance = np.std(distances)
max_distance = np.max(distances)
min_distance = np.min(distances)

# Print out the statistics
print(f"Mean Epipolar Distance: {mean_distance}")
print(f"Standard Deviation of Epipolar Distance: {std_distance}")
print(f"Max Epipolar Distance: {max_distance}")
print(f"Min Epipolar Distance: {min_distance}")

# Filter distances to exclude values outside of 3 standard deviations
distances_filtered = distances[np.abs(distances - mean_distance) <= 3 * std_distance]

# Save the images with keypoints
img1_kp = cv2.drawKeypoints(img1_undistorted, keypoints1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2_undistorted, keypoints2, None, color=(0, 255, 0))

cv2.imwrite("mini_project_2/exercise_9_2_data/frame1_keypoints.png", img1_kp)
cv2.imwrite("mini_project_2/exercise_9_2_data/frame2_keypoints.png", img2_kp)

# Draw matches between the keypoints in img1 and img2
img_matches = cv2.drawMatches(img1_undistorted, keypoints1, img2_undistorted, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save the image with matched keypoints
cv2.imwrite("mini_project_2/exercise_9_2_data/frame_matches.png", img_matches)

# Save the histogram of the epipolar distances (filtered) as a PNG
plt.hist(distances_filtered, bins=20)
plt.xlabel('Epipolar Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Epipolar Distances (Filtered)')

# Save as PNG instead of showing it
plt.savefig("mini_project_2/exercise_9_2_data/epipolar_distance_histogram_filtered.png")

# Decompose the essential matrix to get relative rotation and translation
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# Print the relative motion (rotation and translation)
print("\nRelative Motion:")
print(f"Rotation Matrix (R):\n{R}")
print(f"Translation Vector (t):\n{t}")


################# Exercise 9.3 #################

map = Map()
map.camera_matrix = K

imgs = [img1_undistorted, img2_undistorted]
tfs = [(np.eye(3), np.zeros((3, 1))), (R, t)]
proj_mats = []
cams = []

for i, (img, (R_i, t_i)) in enumerate(zip(imgs, tfs), start=1):
    cam = TrackedCamera(R=R_i, t=t_i, frame_id=i, frame=img, camera_id=i)
    cams.append(cam)
    map.add_camera(cam)
    
    # Compute the projection matrix P = K * [R | t] (4x3)
    # K: intrinsic matrix, R: rotation matrix, t: translation vector
    proj_mats.append(K @ np.hstack((R_i, t_i)))

# Flatten and transpose 2D points for triangulation (shape: 2 x N)
pts_flat = [pts.reshape(-1, 2).T for pts in (pts1, pts2)]

# Triangulate 3D points in homogeneous coordinates (4 x N)
# * unpacks pts_flat into separate arguments, it is NOT a pointer
points_4d = cv2.triangulatePoints(proj_mats[0], proj_mats[1], *pts_flat)
# Convert from homogeneous (x, y, z, w)
# to Euclidean (N, 3) 3D points (x/w, y/w, z/w)
points_3d = (points_4d[:3] / points_4d[3]).T

for i, pt3d in enumerate(points_3d):
    # Create a tracked point and add it to the map
    tracked_point = TrackedPoint(
        point=pt3d,
        descriptor=descriptors1[good_matches[i].queryIdx],
        color=None,
        feature_id=good_matches[i].queryIdx,
        point_id=i + 1
    )
    map.add_point(tracked_point)
    
    # Create and store observations for each camera
    for cam, pts in zip(cams, (pts1, pts2)):
        obs = type('Observation', (), {})()
        obs.camera_id = cam.camera_id
        obs.point_id = tracked_point.point_id
        obs.image_coordinates = pts[i, 0]
        map.observations.append(obs)
        
# Calculate reprojection error
reprojection_errors = []

# Iterate over all the tracked points
for i, pt3d in enumerate(points_3d):
    # Project the 3D point back to image coordinates for both cameras
    for cam, pts in zip(cams, (pts1, pts2)):
        # Get the projection matrix for the current camera
        P = K @ np.hstack((cam.R, cam.t))  # P = K * [R | t]
        
        # Project the 3D point (homogeneous coordinates)
        pt_3d_hom = np.append(pt3d, 1)  # Convert to homogeneous coordinates (x, y, z, 1)
        pt_projected_hom = P @ pt_3d_hom  # Project 3D point

        # Convert back to non-homogeneous image coordinates
        pt_projected = pt_projected_hom[:2] / pt_projected_hom[2]

        # Calculate the reprojection error as the Euclidean distance between projected and observed point
        pt_observed = pts[i, 0]  # Get the observed point from the matched keypoints
        reprojection_error = np.linalg.norm(pt_projected - pt_observed)

        reprojection_errors.append(reprojection_error)

# Calculate summary statistics of the reprojection errors (before filtering)
mean_reprojection_error = np.mean(reprojection_errors)
std_reprojection_error = np.std(reprojection_errors)
max_reprojection_error = np.max(reprojection_errors)
min_reprojection_error = np.min(reprojection_errors)

# Remove everything outside 3 sigma
reprojection_errors = np.array(reprojection_errors)
reprojection_errors = reprojection_errors[np.abs(reprojection_errors - mean_reprojection_error) <= 3 * std_reprojection_error]

mean_reprojection_error = np.mean(reprojection_errors)
std_reprojection_error = np.std(reprojection_errors)
max_reprojection_error = np.max(reprojection_errors)
min_reprojection_error = np.min(reprojection_errors)

# Print out the statistics
print(f"\nReprojection Error:")
print(f"Mean Reprojection Error: {mean_reprojection_error}")
print(f"Standard Deviation of Reprojection Error: {std_reprojection_error}")
print(f"Max Reprojection Error: {max_reprojection_error}")
print(f"Min Reprojection Error: {min_reprojection_error}")

# Calculate the number of bins using Freedman-Diaconis rule
iqr = np.percentile(reprojection_errors, 75) - np.percentile(reprojection_errors, 25)
bin_width = 2 * iqr * len(reprojection_errors) ** (-1/3)
bins = int((reprojection_errors.max() - reprojection_errors.min()) / bin_width)

# Plot the reprojection errors as a density plot
plt.hist(reprojection_errors, bins=bins, density=True, alpha=0.6, color='g')

# Add relevant vertical lines (min, max, mean, std)
plt.axvline(min_reprojection_error, color='c', linestyle='dashed', linewidth=2, label=f'Min: {min_reprojection_error:.2f}')
plt.axvline(mean_reprojection_error - std_reprojection_error, color='b', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {mean_reprojection_error - std_reprojection_error:.2f}')
plt.axvline(mean_reprojection_error, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reprojection_error:.2f}')
plt.axvline(mean_reprojection_error + std_reprojection_error, color='b', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {mean_reprojection_error + std_reprojection_error:.2f}')
plt.axvline(max_reprojection_error, color='y', linestyle='dashed', linewidth=2, label=f'Max: {max_reprojection_error:.2f}')

# Ticks
plt.xticks(np.arange(0, np.ceil(reprojection_errors.max()), 50))

# Plot settings
plt.xlabel('Reprojection Error')
plt.ylabel('Density')
# histogram with outliers outside of 3 sigma removed. lets mention sometihng like gt +3 sigma, lt -3 sigma removed
# plt.title('Histogram of Reprojection Errors (\u00B13\u03C3 removed)')
plt.title('Histogram of Reprojection Errors (Within μ \u00B1 3\u03C3)')

plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
