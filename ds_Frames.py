import numpy as np
import cv2
import os
import sys

vid_folder = sys.argv[1]

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), vid_folder)
sourceDir = os.path.join(root, 'frames')
destDir = os.path.join(root, 'frames_downsampled')

if os.path.isdir(destDir):
    pass
else:
    os.mkdir(destDir)

for file in os.listdir(sourceDir):
    if file.endswith('.png'):
        
        filename = os.path.join(sourceDir, file)
        
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        resized = cv2.resize(np.uint8(img), (64, 36), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(os.path.join(destDir, file), resized)
        
        # cv2.imshow("image", resized)
        # cv2.waitKey(100)
        
        # cv2.destroyAllWindows()