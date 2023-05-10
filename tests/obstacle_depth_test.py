import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys
from obstacle_depth import upperbody_obstacle_crop, extractRegion, show_region, extract_points, obstacle_overlay

# path for recorded bag file
recorded_file = "../Video/extra/20230504_171902.bag"

# initialize bag reader and config
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(recorded_file)
im_rgbd = bag_reader.next_frame()
width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
intrinsic_t = o3d.core.Tensor(cameraMatrix)

# initialize segmentation model
model = ns.seg.init_model()

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



while not bag_reader.is_eof():

    image = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)
    point_cloud_crop = upperbody_obstacle_crop(im_rgbd, intrinsic_t)
    regions = extractRegion(point_cloud_crop)
    image_region = show_region(image, regions)

    cleaned = process1(im_rgbd, cameraMatrix)

    cv2.imshow('cleaned', cleaned)
    cv2.imshow('rw', image)
    cv2.imshow('regions', image_region)
    
    img = np.zeros((500,500,3))
    ob_points = extract_points(regions)

    overlayed = obstacle_overlay(cleaned, ob_points)
    cv2.imshow("overlayed", overlayed)
   
    key = cv2.waitKey(1)
        # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
    im_rgbd = bag_reader.next_frame()

bag_reader.close()