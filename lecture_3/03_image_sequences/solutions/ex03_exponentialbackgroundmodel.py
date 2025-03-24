import numpy as np
import cv2
from lsdp_tools import FrameIterator
import matplotlib.pyplot as plt


class MotionDetectionMeanBackgroundModel:
    def __init__(self):
        self.generator = FrameIterator('../input/remember.mkv').frame_generator()
        self.counter = 0
        self.accumulator = None
        self.decay_factor = 0.98
        self.mean_image = None
        self.open_and_place_windows()


    def open_and_place_windows(self):
        # Open and place windows
        cv2.namedWindow("frame")
        cv2.namedWindow("mean_image")
        cv2.namedWindow("diff_image")
        cv2.namedWindow("motion")
        cv2.moveWindow("mean_image", 650, 0)
        cv2.moveWindow("frame", 0, 0)
        cv2.moveWindow("diff_image", 0, 500)
        cv2.moveWindow("motion", 650, 500)


    def update_background_model(self):
        # Initialize the accumulator
        if self.accumulator is None:
            self.accumulator = self.frame.astype(float)

        # Calculate mean value of all pixels 
        self.accumulator = (1 - self.decay_factor) * self.frame + self.decay_factor * self.accumulator
        self.mean_image = self.accumulator
        cv2.imshow("mean_image", self.mean_image / 255)


    def detect_motion(self):
        # Detect motion using your model.
        diff_image = self.frame - self.mean_image
        cv2.imshow("diff_image", 0.5 * diff_image / 255 + 0.5)
        test, changes = cv2.threshold(np.abs(diff_image), 50, 255, cv2.THRESH_BINARY)
        cv2.imshow("motion", changes)


    def main(self):
        # Look at each frame in the video one at a time.
        for frame in self.generator:
            self.frame = frame
            self.counter += 1

            self.update_background_model()
            self.detect_motion()

            cv2.imshow("frame", self.frame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break


model = MotionDetectionMeanBackgroundModel()
model.main()
