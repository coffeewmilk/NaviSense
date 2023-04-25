import _init_paths
import NaviSense as ns
import numpy as np
import cv2


#import image
image_path = './tests/data/0.jpg'
img = cv2.imread(image_path)


# initialize segmentation model
model = ns.seg.init_model()
segmented = ns.seg.segmentationfun(model, img)

cv2.imshow('segmented', segmented)

key = cv2.waitKey(0)
# if pressed escape exit program
if key == 27:
    cv2.destroyAllWindows()


