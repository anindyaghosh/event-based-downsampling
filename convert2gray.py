import cv2
import os
import sys

vid_folder = sys.argv[1]

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), vid_folder)

for file in os.listdir(root):
    if file.endswith('.png'):
        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
        
        filehead, ext = file.split('.')
        
        cv2.imwrite(os.path.join(root, '_'.join([filehead, 'gray']) + '.' + ext), img)

        # cv2.imshow('Image', img)
        
        # cv2.waitKey(0)
        
        # cv2.destroyAllWindows()