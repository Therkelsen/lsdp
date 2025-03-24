#!/usr/bin/python
# -*- coding: utf8 -*-
import cv2
import numpy as np
from lsdp_tools import FrameIterator

class DenseOpticalFlow():
    def __init__(self):
        self.generator = FrameIterator('../input/remember.mkv').frame_generator()
        self.motion_threshold = 20


    def find_contours_of_moving_objects(self, prvs, prvs2, next_frame):
        """Locate moving objects with a certain size and return their contours."""
        self.moving_elements = 0.5 * (cv2.absdiff(prvs, next_frame) >
                self.motion_threshold) + 0.5 * (cv2.absdiff(prvs2, next_frame) >
                        self.motion_threshold)
        self.moving_elements = cv2.GaussianBlur(self.moving_elements, (5,5), 0)
        ret, thresh = cv2.threshold(255*self.moving_elements,127,255,0)
        thresh = np.array(thresh, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


    def draw_circles_on_contours(self, frame, contours):
        """Draw circles on the center of mass of a set of contours."""
        for cnt in contours:
            M = cv2.moments(cnt)
            if(M['m00'] > 100):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(frame, (cx, cy), 20, (255, 0, 0), 2)
        return frame


    def main(self):
        frame = next(self.generator)
        prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        prvs2 = prvs
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        optical_flow.setGamma(1)

        for frame in self.generator:
            # Skip every second frame
            for k in range(2):
                continue

            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contours = self.find_contours_of_moving_objects(prvs, prvs2,
                    next_frame)
            frame = self.draw_circles_on_contours(frame, contours)
                
            cv2.imshow('frame', frame)
            cv2.imshow('framediff', self.moving_elements)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
            elif k == ord('p'):
                cv2.waitKey(1000000)
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame)
            prvs2 = prvs
            prvs = next_frame

        self.cap.release()
        cv2.destroyAllWindows()

temp = DenseOpticalFlow()
temp.main()

