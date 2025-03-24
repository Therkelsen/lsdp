import numpy as np
import cv2


class FrameIterator():
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)

    def frame_generator(self):
        # Define a generator that yields frames from the video
        while(1):
            ret, frame = self.cap.read()
            if ret is not True:
                break
            yield frame
        self.cap.release()

    def main(self):
        for frame in self.frame_generator():
            # Process frame
            cv2.imshow('frame',frame)

            # Deal with key presses
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite("../output/ex00_stillimage.png", frame)

fi = FrameIterator('../input/remember.mkv')
fi.main()
