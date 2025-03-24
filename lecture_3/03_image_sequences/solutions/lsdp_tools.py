import cv2

class FrameIterator():
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        assert self.cap.isOpened(), "Could not open video stream."
        
    def frame_generator(self):
        # Define a generator that yields frames from the video
        while(1):
            ret, frame = self.cap.read()
            if ret is not True:
                break
            yield frame
        self.cap.release()


