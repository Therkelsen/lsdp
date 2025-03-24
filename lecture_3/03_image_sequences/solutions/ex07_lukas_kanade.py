import numpy as np
import cv2
from lsdp_tools import FrameIterator
import matplotlib.pyplot as plt


class MotionDetectionMeanBackgroundModel:
    def __init__(self):
        self.generator = FrameIterator('../input/remember.mkv').frame_generator()
        self.open_and_place_windows()

        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=500,
                              qualityLevel=0.05,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0, 255, (1000, 3))


    def open_and_place_windows(self):
        # Open and place windows
        cv2.namedWindow("frame")
        cv2.moveWindow("frame", 0, 0)


    def main(self):
        # Look at each frame in the video one at a time.
        first_frame = next(self.generator)

        # Take first frame and find corners in it
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Locate good features to track
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(first_frame)

        for frame in self.generator:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0,
                    None, **self.lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(img=mask,
                                pt1=(int(a), int(b)),
                                pt2=(int(c), int(d)),
                                color=self.color[i].tolist(),
                                thickness=2)
                frame = cv2.circle(img=frame,
                                   center=(int(a), int(b)),
                                   radius=0,
                                   color=self.color[i].tolist(),
                                   thickness=-1)

            img = cv2.add(frame, mask)
            mask = cv2.addWeighted(mask, 0.99, mask, 0, 0)
            cv2.imshow('frame', img)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
            if k == ord('r'):
                # Reset points to track
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)


model = MotionDetectionMeanBackgroundModel()
model.main()
