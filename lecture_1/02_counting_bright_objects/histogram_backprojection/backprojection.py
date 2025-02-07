import numpy as np
import cv2

# Based on example from
# https://docs.opencv.org/3.4/dc/df6/tutorial_py_histogram_backprojection.html

# Load image with examples of the color that should be
# located later.
reference_image = cv2.imread('input/grass.jpg')
ref_im_in_hsv = cv2.cvtColor(reference_image,
                             cv2.COLOR_BGR2HSV)


# Calculate color histogram (only for the H and S values).
ref_im_histogram = cv2.calcHist(images=[ref_im_in_hsv],
                                channels=[0, 1],
                                mask=None,
                                histSize=[180, 256],
                                ranges=[0, 180, 0, 256])

# Normalize histogram
cv2.normalize(src=ref_im_histogram,
              dst=ref_im_histogram,
              alpha=0,
              beta=255,
              norm_type=cv2.NORM_MINMAX)
reference_histogram_in_bgr = cv2.merge((ref_im_histogram,
                                        ref_im_histogram,
                                        ref_im_histogram))

# Save histogram and inverted histogram
cv2.imwrite('output/00_reference_HS_histogram.png',
            reference_histogram_in_bgr)
cv2.imwrite('output/00_reference_HS_histogram_inverted.png',
            255 - reference_histogram_in_bgr)

# Load the image that should be analysed / segmented.
new_image = cv2.imread('input/messi.jpg')
new_image_in_hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)

# Apply backprojection.
backproject_image = cv2.calcBackProject(images=[new_image_in_hsv],
                                        channels=[0, 1],
                                        hist=ref_im_histogram,
                                        ranges=[0, 180, 0, 256],
                                        scale=1)
cv2.imwrite('output/10_backprojected_image.png', backproject_image)

# Enlarge the located regions that contain the
# reference color.
# Do this using a dilation with a 5x5 ellipse.
disc_se = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                    ksize=(5, 5))
backproject_image = cv2.filter2D(src=backproject_image,
                                 ddepth=-1,
                                 kernel=disc_se)
cv2.imwrite('output/15_backprojected_image_dilated.png', backproject_image)

# Make a mask image in the BGR format by replicating the threshold image.
ret, backproject_threshold = cv2.threshold(src=backproject_image,
                                           thresh=50,
                                           maxval=255,
                                           type=cv2.THRESH_BINARY)

backproject_threshold = cv2.merge(
    (backproject_threshold,
     backproject_threshold,
     backproject_threshold))

# Color all pixels black that does not match the reference color.
res = cv2.bitwise_and(new_image, backproject_threshold)

# Stack the three images on top of each other.
res = np.vstack((new_image, backproject_threshold, res))
cv2.imwrite('output/20_combined_result.jpg', res)
