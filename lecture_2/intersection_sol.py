from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from lxml import etree
import argparse


class Image:
    def __init__(self, line, focal):
        elements = line.split('\t')
        self.ID = elements[0] + '.JPG'
        self.X = float(elements[1])
        self.Y = float(elements[2])
        self.Z = float(elements[3])
        self.R = np.array([[float(elements[4]), float(elements[5]), float(elements[6])],
                           [float(elements[7]), float(elements[8]), float(elements[9])],
                           [float(elements[10]), float(elements[11]), float(elements[12])]])
        self.f = focal


def images_import(fid, focal):
    with open(fid) as f:
        for noElements, l in enumerate(f):
            pass
    ImageSet = {}
    with open(fid) as f:
        f.readline()
        for i in range(0, noElements):
            p = f.readline()
            image = Image(p, focal)
            ImageSet[image.ID] = image
    return ImageSet


def reprojection(image, point):
    # reproject point on the image
    x = -image.f * \
        (image.R[0, 0] * (point[0] - image.X) + image.R[1, 0] * (point[1] - image.Y) + image.R[2, 0] * (point[2] - image.Z)) / \
        (image.R[0, 2] * (point[0] - image.X) + image.R[1, 2] * (point[1] - image.Y) + image.R[2, 2] * (point[2] - image.Z))
    y = -image.f * \
        (image.R[0, 1] * (point[0] - image.X) + image.R[1, 1] * (point[1] - image.Y) + image.R[2, 1] * (point[2] - image.Z)) / \
        (image.R[0, 2] * (point[0] - image.X) + image.R[1, 2] * (point[1] - image.Y) + image.R[2, 2] * (point[2] - image.Z))

    return [x, y]


def intersection_multiple(left_image, right_image, xy_left_focal, xy_right_focal):
    # intersection of two rays (left and right image ray) to calculate 3D pint coordinates
    # equation ATA-1*ATL so according to LSM
    EOE_l = [left_image.X, left_image.Y, left_image.Z]
    EOE_r = [right_image.X, right_image.Y, right_image.Z]
    R_l = left_image.R
    R_r = right_image.R

    # calculate all A arrays
    A = np.zeros([xy_left_focal.shape[0], 12])
    A[:, 0] = R_l[0, 2] * xy_left_focal[:, 0].T + R_l[0, 0] * left_image.f
    A[:, 1] = R_l[1, 2] * xy_left_focal[:, 0].T + R_l[1, 0] * left_image.f
    A[:, 2] = R_l[2, 2] * xy_left_focal[:, 0].T + R_l[2, 0] * left_image.f
    A[:, 3] = R_l[0, 2] * xy_left_focal[:, 1].T + R_l[0, 1] * left_image.f
    A[:, 4] = R_l[1, 2] * xy_left_focal[:, 1].T + R_l[1, 1] * left_image.f
    A[:, 5] = R_l[2, 2] * xy_left_focal[:, 1].T + R_l[2, 1] * left_image.f
    A[:, 6] = R_r[0, 2] * xy_right_focal[:, 0].T + R_r[0, 0] * right_image.f
    A[:, 7] = R_r[1, 2] * xy_right_focal[:, 0].T + R_r[1, 0] * right_image.f
    A[:, 8] = R_r[2, 2] * xy_right_focal[:, 0].T + R_r[2, 0] * right_image.f
    A[:, 9] = R_r[0, 2] * xy_right_focal[:, 1].T + R_r[0, 1] * right_image.f
    A[:, 10] = R_r[1, 2] * xy_right_focal[:, 1].T + R_r[1, 1] * right_image.f
    A[:, 11] = R_r[2, 2] * xy_right_focal[:, 1].T + R_r[2, 1] * right_image.f

    # calculate all L arrays
    L = np.zeros((4, xy_left_focal.shape[0]))
    L[0, :] = left_image.f * R_l[0, 0] * EOE_l[0] \
              + left_image.f * R_l[1, 0] * EOE_l[1] \
              + left_image.f * R_l[2, 0] * EOE_l[2] \
              + xy_left_focal[:, 0].T * R_l[0, 2] * EOE_l[0] \
              + xy_left_focal[:, 0].T * R_l[1, 2] * EOE_l[1] \
              + xy_left_focal[:, 0].T * R_l[2, 2] * EOE_l[2]
    L[1, :] = left_image.f * R_l[0, 1] * EOE_l[0] \
              + left_image.f * R_l[1, 1] * EOE_l[1] \
              + left_image.f * R_l[2, 1] * EOE_l[2] \
              + xy_left_focal[:, 1].T * R_l[0, 2] * EOE_l[0] \
              + xy_left_focal[:, 1].T * R_l[1, 2] * EOE_l[1] \
              + xy_left_focal[:, 1].T * R_l[2, 2] * EOE_l[2]
    L[2, :] = right_image.f * R_r[0, 0] * EOE_r[0] \
              + right_image.f * R_r[1, 0] * EOE_r[1] \
              + right_image.f * R_r[2, 0] * EOE_r[2] \
              + xy_right_focal[:, 0].T * R_r[0, 2] * EOE_r[0] \
              + xy_right_focal[:, 0].T * R_r[1, 2] * EOE_r[1] \
              + xy_right_focal[:, 0].T * R_r[2, 2] * EOE_r[2]
    L[3, :] = right_image.f * R_r[0, 1] * EOE_r[0] \
              + right_image.f * R_r[1, 1] * EOE_r[1] \
              + right_image.f * R_r[2, 1] * EOE_r[2] \
              + xy_right_focal[:, 1].T * R_r[0, 2] * EOE_r[0] \
              + xy_right_focal[:, 1].T * R_r[1, 2] * EOE_r[1] \
              + xy_right_focal[:, 1].T * R_r[2, 2] * EOE_r[2]

    # perform intersection
    XYZ = np.zeros((xy_left_focal.shape[0], 3))
    for i in range(0, xy_left_focal.shape[0]):
        A_temp = np.asarray([[A[i, 0], A[i, 1], A[i, 2]],
                             [A[i, 3], A[i, 4], A[i, 5]],
                             [A[i, 6], A[i, 7], A[i, 8]],
                             [A[i, 9], A[i, 10], A[i, 11]]])
        L_temp = L[:, i]
        AT = A_temp.T
        ATA = np.matmul(AT, A_temp)
        ATL = np.matmul(AT, L_temp)
        x = np.matmul(np.linalg.inv(ATA), ATL)
        v = np.matmul(A_temp, x) - L_temp
        RMSE = np.matmul(v.T, v) ** 0.5
        print('RMSE', RMSE)
        XYZ[i, :] = np.matmul(np.linalg.inv(ATA), ATL)

    return XYZ


def plt_approximation(event, x, y, flags, param):
    # # grab references to the global variables
    global ref_pt
    if event == cv.EVENT_LBUTTONDOWN:
        ref_pt.append((x, y))


# ---------------------------------------------------------------------------------------------------------------------


# load images
parser = argparse.ArgumentParser(description='Calculate intersection to get 3D point position.')
parser.add_argument('--extrinsic', type=str, help='file containing extrinsic parameters')
parser.add_argument('--intrinsic', type=str, help='file containing camera calibration')
parser.add_argument('--img', type=str, help='path to images')
parser.add_argument('--left', type=str, help='left image')
parser.add_argument('--right', type=str, help='right image')

args = parser.parse_args()

camera_calib = etree.parse(args.intrinsic)
focal = float(camera_calib.find('f').text)
image_dict = images_import(args.extrinsic, focal)

left_raw = args.left
right_raw = args.right

left = args.img + left_raw
right = args.img + right_raw

img_l = cv.imread(left)
img_r = cv.imread(right)

plt.subplot(121), plt.imshow(cv.cvtColor(img_l, cv.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv.cvtColor(img_r, cv.COLOR_BGR2RGB))
plt.show()

# get points positions for the left image

ref_pt = []

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", plt_approximation)
cv.imshow("image", img_l)
cv.waitKey(0)
cv.destroyAllWindows()
left_pix = np.asarray(ref_pt)

# get points positions for the right image

ref_pt = []

cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.setMouseCallback("image", plt_approximation)
cv.imshow("image", img_r)
cv.waitKey(0)
cv.destroyAllWindows()
right_pix = np.asarray(ref_pt)

# calculate intersection

height = img_l.shape[0]
width = img_l.shape[1]

left_focal = np.zeros(left_pix.shape)
right_focal = np.zeros(right_pix.shape)
left_focal[:, 0] = left_pix[:, 0] - width / 2
left_focal[:, 1] = -left_pix[:, 1] + height / 2
right_focal[:, 0] = right_pix[:, 0] - width / 2
right_focal[:, 1] = -right_pix[:, 1] + height / 2

XYZ = intersection_multiple(image_dict[left_raw], image_dict[right_raw], left_focal, right_focal)

print('left image position:')
print(image_dict[left_raw].X, image_dict[left_raw].Y, image_dict[left_raw].Z)
print('right image position:')
print(image_dict[right_raw].X, image_dict[right_raw].Y, image_dict[right_raw].Z)
print('Intersection results for all of the points:')
print(XYZ)

# check results
vxvy_l = np.zeros((XYZ.shape[0], 2))
vxvy_r = np.zeros((XYZ.shape[0], 2))

for i, point in enumerate(XYZ):
    left_point = np.asarray(reprojection(image_dict[left_raw], point))
    right_point = np.asarray(reprojection(image_dict[right_raw], point))
    vxvy_l[i, :] = left_focal[i, :] - left_point
    vxvy_r[i, :] = right_focal[i, :] - right_point

print('reprojection disparity for left image')
print(vxvy_l)

print('reprojection disparity for right image')
print(vxvy_r)
