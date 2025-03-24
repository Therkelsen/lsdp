import numpy as np
import cv2
from lsdp_tools import FrameIterator

def main():
    fi = FrameIterator('../input/remember.mkv')
    generator = fi.frame_generator()

    # These are the determined sample locations.
    # The purpose of the script is to visualize their position 
    # on top of the video, this will enable us to choose proper 
    # sample locations for the next exercise.
    foreground = np.array([[177, 73], [256, 146]])
    background = np.array([[456, 378], [581, 306], [125, 278]])

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org 
    org = (50, 50) 
      
    # fontScale 
    fontScale = 1
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 2
       
    counter = 0
    for frame in generator:
        counter += 1
        
        for point in background: 
            cv2.circle(frame, tuple(point), 5, (0, 255, 255), 3)
        for point in foreground: 
            cv2.circle(frame, tuple(point), 5, (255, 255, 0), 3)

        # Using cv2.putText() method 
        frame = cv2.putText(frame, "%d" % counter, org, font,  
                           fontScale, color, thickness, cv2.LINE_AA) 

        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite("../output/ex01stillimage.png", frame)
    cv2.destroyAllWindows()


main()
