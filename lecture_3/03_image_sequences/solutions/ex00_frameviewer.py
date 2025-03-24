import cv2
from lsdp_tools import FrameIterator


def main():
    fi = FrameIterator('../input/remember.mkv')
    generator = fi.frame_generator()

    # Access first frame
    frame = next(generator)
    cv2.imshow("test", frame)
    cv2.waitKey(100000)


    # Iterate over the remaining frames
    for frame in generator:
        cv2.imshow("test", frame)

        # Process frame
        cv2.imshow('frame',frame)

        # Deal with key presses
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite("../output/ex00_stillimage.png", frame)


main()

