import cv2
import numpy as np


def count_fully_saturated_pixels(img):
    mask = cv2.inRange(img, (255, 255, 255), (255, 255, 255))
    count = np.count_nonzero(mask)
    return count


def count_red_saturated_pixels(img):
    mask = cv2.inRange(img, (0, 0, 255), (255, 255, 255))
    count = np.count_nonzero(mask)
    return count


def count_green_saturated_pixels(img):
    mask = cv2.inRange(img, (0, 255, 0), (255, 255, 255))
    count = np.count_nonzero(mask)
    return count


def count_blue_saturated_pixels(img):
    mask = cv2.inRange(img, (255, 0, 0), (255, 255, 255))
    count = np.count_nonzero(mask)
    return count


def count_partially_saturated_pixels(img):
    maskb = cv2.inRange(img, (255, 0, 0), (255, 255, 255))
    maskg = cv2.inRange(img, (0, 255, 0), (255, 255, 255))
    maskr = cv2.inRange(img, (0, 0, 255), (255, 255, 255))
    partially_saturated = np.logical_or(maskb, maskg, maskr)
    count = np.count_nonzero(partially_saturated)
    return count


def analyse_saturated_pixels(filename):
    print("Analysing file: ", filename)
    img = cv2.imread(filename)
    print("Partial saturated pixels: %7d" % count_partially_saturated_pixels(img))
    print("Fully saturared pixels:   %7d" % count_fully_saturated_pixels(img))
    print("Red saturated pixels:     %7d" % count_red_saturated_pixels(img))
    print("Green saturated pixels:   %7d" % count_green_saturated_pixels(img))
    print("Blue saturated pixels:    %7d" % count_blue_saturated_pixels(img))


analyse_saturated_pixels('../input/over_exposed_DJI_0215.JPG')
analyse_saturated_pixels('../input/well_exposed_DJI_0214.JPG')
analyse_saturated_pixels('../input/under_exposed_DJI_0213.JPG')
