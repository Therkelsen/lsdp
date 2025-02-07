import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Calculations all take place in the Universal Transverse Mercator (UTM)
# coordinate system \addref{}, where $x$ is displacement towards east,
# $y$ is displacement towards north and $z$ is the altitude above sea level.

# The orientation of the camera is described by
# the three rotation angles: $yaw$, $pitch$ and $roll$.
# Assuming that the camera is pointing towards the horizon,
# the \emph{yaw} values is the compass heading that the camera is looking towards,
# e.g. $yaw = 0\degree$ is towards north and $yaw = 45\degree$ is towards north east.
# The camera $pitch$ is the angle of the camera relative to the horizon,
# when $pitch = 0\degree$ the camera is looking directly forward and when
# $pitch = -90\degree$ the camera is looking directly towards the ground.
# The $roll$ values is how much the horizon is away from being horizontal;
# this will in practice often be zero as the camera gimbal ensures that.


def get_yaw_matrix(yaw_angle):
    return np.array([[np.cos(yaw_angle), np.sin(yaw_angle), 0],
                     [-np.sin(yaw_angle), np.cos(yaw_angle), 0],
                     [0, 0, 1]])


def get_pitch_matrix(pitch_angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
                     [0, np.sin(pitch_angle), np.cos(pitch_angle)]])


def get_roll_matrix(roll_angle):
    return np.array([[np.cos(roll_angle), 0, np.sin(roll_angle)],
                     [0, 1, 0],
                     [-np.sin(roll_angle), 0, np.cos(roll_angle)]])


# Yaw is rotation around the $z$ axis (up / down)
yaw_matrix = get_yaw_matrix(yaw_angle=10 * np.pi / 180)

# Pitch is rotation around the x axis (right / left)
pitch_matrix = get_pitch_matrix(pitch_angle=-40 * np.pi / 180)

# Roll is rotation around the y axis (forward / back)
roll_matrix = get_roll_matrix(roll_angle=4 * np.pi / 180)

R_pose = np.matmul(yaw_matrix, np.matmul(pitch_matrix, roll_matrix))

# print(combined_pose_matrix)
hfov = 41.22 * np.pi / 180
vfov = 70.67 * np.pi / 180
im_points = np.array([[-np.tan(vfov / 2), 1, -np.tan(hfov / 2)],
                      [-np.tan(vfov / 2), 1, np.tan(hfov / 2)],
                      [np.tan(vfov / 2), 1, np.tan(hfov / 2)],
                      [np.tan(vfov / 2), 1, -np.tan(hfov / 2)],
                      [-np.tan(vfov / 2), 1, -np.tan(hfov / 2)]])
im_points = np.transpose(im_points)


# Show the image plane in front of the camera
# ===========================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Image plane in front of camera')
# Camera
ax.scatter(0, 0, 0)

# Viewport in front of camera
ax.plot(im_points[0, :],
        im_points[1, :],
        im_points[2, :],
        color='red')

ax.set_xlabel('Right / Left (x)')
ax.set_ylabel('Forward / Backward (y)')
ax.set_zlabel('Up / down (z)')

ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
plt.savefig("output/01imageplane.png")
plt.show()

# Press q to close the plot


# Rotate the viewpoint according to the yaw rotation matrix
# =========================================================
yaw_corrected_points = np.matmul(yaw_matrix, im_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Correcting yaw')
# Camera
ax.scatter(0, 0, 0)

# Viewport in front of camera
ax.plot(im_points[0, :],
        im_points[1, :],
        im_points[2, :],
        color='red')

# Viewport in front of camera
ax.plot(yaw_corrected_points[0, :],
        yaw_corrected_points[1, :],
        yaw_corrected_points[2, :],
        color='blue')

ax.set_xlabel('Right / Left (x)')
ax.set_ylabel('Forward / Backward (y)')
ax.set_zlabel('Up / down (z)')

ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
plt.savefig("output/02correctyaw.png")
plt.show()


# Rotate the viewpoint according to the full pose matrix
# ======================================================
directions = np.matmul(R_pose, im_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Full camera rotation (yaw, pitch, roll)')
# Camera
ax.scatter(0, 0, 0)

# Viewport in front of camera
ax.plot(im_points[0, :],
        im_points[1, :],
        im_points[2, :],
        color='red')

# Rotated viewport
ax.plot(directions[0, :],
        directions[1, :],
        directions[2, :],
        color='blue')

ax.set_xlabel('Right / Left (x)')
ax.set_ylabel('Forward / Backward (y)')
ax.set_zlabel('Up / down (z)')

ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
plt.savefig("output/03fullrotationcorrection.png")
plt.show()



# Project image plane on the ground
altitude = 2
down = np.array([[0, 0, -1]])

z_components = np.matmul(down, directions)
scaling_factors = altitude / z_components
scaling_factor_matrix = np.diag(scaling_factors[0])
corners_on_ground = np.matmul(directions, scaling_factor_matrix)

with np.printoptions(precision=3, suppress=True):
    print("yaw_matrix")
    print(yaw_matrix)
    print("pitch_matrix")
    print(pitch_matrix)
    print("roll_matrix")
    print(roll_matrix)
    print("R_pose")
    print(R_pose)
    print("im_points:")
    print(im_points)
    print("Directions:")
    print(directions)
    print("z_components:")
    print(z_components)
    print("scaling_factors:")
    print(scaling_factors)
    print("scaling_factor_matrix")
    print(scaling_factor_matrix)
    print("corners_on_ground")
    print(corners_on_ground)


# Extend the vectors to the ground plane
# ======================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Extend vectors to ground plane')
# Camera
ax.scatter(0, 0, 0)

# Viewport in front of camera
ax.plot(im_points[0, :],
        im_points[1, :],
        im_points[2, :],
        color='red')

# Rotated viewport
ax.plot(directions[0, :],
        directions[1, :],
        directions[2, :],
        color='blue')

# Rotated viewport
ax.plot(corners_on_ground[0, :],
        corners_on_ground[1, :],
        corners_on_ground[2, :],
        color='black')

ax.set_xlabel('Right / Left (x)')
ax.set_ylabel('Forward / Backward (y)')
ax.set_zlabel('Up / down (z)')

ax.auto_scale_xyz([-3, 3], [0, 6], [-3, 3])
plt.savefig("output/04projecttoground.png")
plt.show()
