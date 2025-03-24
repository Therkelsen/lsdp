import numpy as np
import cv2 as cv
from lsdp_tools import FrameIterator

fi = FrameIterator('lecture_3/03_image_sequences/input/Sometimes Security Cameras catch a gem.mp4')
generator = fi.frame_generator()

bg = np.array([[273, 395], [600, 311]])
fg = np.array([[156, 102], [388, 248]])

# font 
font = cv.FONT_HERSHEY_SIMPLEX 
    
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
    
    for point in bg: 
        cv.circle(frame, tuple(point), 5, (0, 255, 255), 3)
    for point in fg: 
        cv.circle(frame, tuple(point), 5, (255, 255, 0), 3)

    # Using cv.putText() method 
    frame = cv.putText(frame, "%d" % counter, org, font,  
                        fontScale, color, thickness, cv.LINE_AA) 

    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite("../output/ex01stillimage.png", frame)

cv.destroyAllWindows()
