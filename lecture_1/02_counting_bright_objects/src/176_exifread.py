import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import exifread
from xml.dom.minidom import parseString

def compare_original_and_segmented_image(original, segmented, title):
    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(1, 2, 1)
    plt.title(title)
    ax1.imshow(original)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(segmented)

def get_gimbal_orientation(filename):
    yaw = None
    pitch = None
    roll = None

    # Open the file and extract exif information.
    f = open(filename, 'rb')
    # The debug = True option is needed to search for xmp information
    # in the .jgp file. 
    tags = exifread.process_file(f, debug=True)

    if "Image ApplicationNotes" in tags.keys():
        # Extract the xmp values and put them in a dictionary.
        dom = parseString(tags["Image ApplicationNotes"].printable)
        temp = dom.getElementsByTagName("rdf:Description")[0].attributes.items()
        attrs = dict(temp)
        # print(attrs)

        # Extract the needed information from the dictionary.
        yaw = float(attrs['drone-dji:GimbalYawDegree'])
        pitch = float(attrs['drone-dji:GimbalPitchDegree'])
        roll = float(attrs['drone-dji:GimbalRollDegree'])
    else:
        raise Exception("Could not find gimbal orientation information")

    return (yaw, pitch, roll)

if __name__ == "__main__":
    input_path = "lecture_1/02_counting_bright_objects/input/"
    output_path = "lecture_1/01_getting_started/output/175/"
    # output_path_image = output_path + "flower_output_gray.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = input_path + "DJI_0222.JPG"
    
    print(get_gimbal_orientation(img))
    