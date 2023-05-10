import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys
import multiprocessing
import time 

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

def process2(queue_in, queue_out):
    global intrinsic_t

    while True:
        im_rgbd = queue_in.get()
        image = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)

        #crop point cloud
        point_cloud_crop = ns.depthbased.upperbody_obstacle_crop(im_rgbd, intrinsic_t)

        #extract cluster
        regions = ns.depthbased.extractRegion(point_cloud_crop)
        #image_region = ns.depthbased.show_region(image, regions)

        #extract points for overlay
        ob_points = ns.depthbased.extract_points(regions)
        queue_out.put((ob_points))


if __name__ == "__main__":

    queue_in = multiprocessing.Queue()
    queue_out = multiprocessing.Queue()
    process2p = multiprocessing.Process(target=process2, args=(queue_in, queue_out))
    process2p.start()


    while not bag_reader.is_eof():

        time1 = time.time()

        # put data in queue
        queue_in.put(im_rgbd)
        
        image = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)
        cleaned = process1(im_rgbd, cameraMatrix)

        cv2.imshow('cleaned', cleaned)
        cv2.imshow('rw', image)
        
        
        img = np.zeros((500,500,3))

        # get value from the process here
        ob_points = queue_out.get()
        
        overlayed = ns.depthbased.obstacle_overlay(cleaned, ob_points)
        cv2.imshow("overlayed", overlayed)

        map = ns.create_occupacny_map(overlayed)
        result = ns.hybrid_maximum_angle(map, overlayed)
        line_result = ns.hybrid_drawLine(overlayed, result)
        cv2.imshow("line", line_result)
        
        
        key = cv2.waitKey(1)
            # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        im_rgbd = bag_reader.next_frame()

        time2 = time.time()
        print(f'Time taken = {time2-time1}')

    bag_reader.close()

    #terminate process
    process2p.terminate()
    process2p.join()