import numpy as np
import cv2
import matplotlib.pyplot as plt
from lsdp_tools import FrameIterator


class MotionDetectionMeanBackgroundModel:
    def __init__(self):
        self.generator = FrameIterator('../input/remember.mkv').frame_generator()
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.background_subtractor = cv2.createBackgroundSubtractorKNN()
        self.open_and_place_windows()


    def open_and_place_windows(self):
        # Open and place windows
        cv2.namedWindow("frame")
        cv2.namedWindow("motion")
        cv2.moveWindow("frame", 0, 0)
        cv2.moveWindow("motion", 650, 0)


    def detect_motion(self):
        res = self.background_subtractor.apply(self.frame)
        cv2.imshow("motion", res)


    def main(self):
        # Look at each frame in the video one at a time.
        for frame in self.generator:
            self.frame = frame
            self.detect_motion()

            cv2.imshow("frame", self.frame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break


model = MotionDetectionMeanBackgroundModel()
model.main()
