#!/usr/bin/python
# -*- coding: utf8 -*-
import cv2
import numpy as np
from lsdp_tools import FrameIterator

class DenseOpticalFlow():
    def __init__(self):
        self.generator = FrameIterator('../input/remember.mkv').frame_generator()
        self.motion_threshold = 20


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
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255
        dense_inverse_search_flow = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow_PRESET_MEDIUM)

        for frame in self.generator:
            # Skip every second frame
            for k in range(2):
                continue

            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = dense_inverse_search_flow.calc(prvs, next_frame, None)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            #hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[...,2] = cv2.min(mag, 255  / 100)  * 100
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
            cv2.imshow('frame', frame)
            cv2.imshow('frame_dis', bgr)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
            elif k == ord('p'):
                cv2.waitKey(1000000)
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame)
                cv2.imwrite('opticalhsv.png', bgr)
            prvs = next_frame

        cv2.destroyAllWindows()

temp = DenseOpticalFlow()
temp.main()

