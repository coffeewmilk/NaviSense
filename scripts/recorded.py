import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys


def process1(im_rgbd, cameraMatrix):
    global model
    
    # detect floor plane
    point2d = ns.getPlane(im_rgbd, cameraMatrix)

    # image segmentation
    raw_image = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)
    segmented = ns.seg.segmentationfun(model, raw_image)

    # transfrom to top view
    wrapped, masked = ns.getTransform(point2d, segmented)

    # cleanup some artifacts
    cleaned = ns.cleanup(wrapped)
    return cleaned



# path for recorded bag file
recorded_file = "../Video/extra/20230329_170533.bag"

# initialize bag reader and config
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(recorded_file)
im_rgbd = bag_reader.next_frame()
width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
intrinsic_t = o3d.core.Tensor(cameraMatrix)

# initialize segmentation model
model = ns.seg.init_model()



while not bag_reader.is_eof():

    #segmented and cleaned
    cleaned = process1(im_rgbd, cameraMatrix)

    #cv2.imshow("Mask", masked)
    #cv2.imshow("wrappped", wrapped)
    cv2.imshow("Cleanuped", cleaned)


    map = ns.create_occupacny_map(cleaned)
    angle, value = ns.max_value_angle(map)
    print(angle)


    cleaned_line = ns.drawLine(cleaned, angle)
    cv2.imshow("line", cleaned_line)
   
    key = cv2.waitKey(1)
        # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
    im_rgbd = bag_reader.next_frame()

bag_reader.close()