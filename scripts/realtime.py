import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys
import json
import multiprocessing


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


if __name__ == "__main__":
    # camera parameters
    width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
    cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    intrinsic_t = o3d.core.Tensor(cameraMatrix)

    # initialize segmentation model
    model = ns.seg.init_model()

    # initialize feedback process and value
    angle = multiprocessing.Value('i', 90)
    sound_p = multiprocessing.Process(target=ns.metronome, args=(angle,))
    sound_p.start()

    # initialize and config camera
    with open('./Scripts/rs_config.json') as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(False)  # true: start recording with capture

    while(True):
        
        # capture data and align them
        im_rgbd = rs.capture_frame(True, True)  
        cleaned = process1(im_rgbd, cameraMatrix)
        cv2.imshow("Cleanuped", cleaned)

        #duplicate 
        cleaned_copy = np.copy(cleaned)


        map = ns.create_occupacny_map(cleaned)
        angle.value, value = ns.max_value_angle(map)
        print(angle)


        cleaned_line = ns.drawLine(cleaned, angle.value)
        cv2.imshow("line", cleaned_line)

        #test new algorithm
        #result = ns.hybrid_maximum_angle(map, cleaned_copy)
        #line_result = ns.hybrid_drawLine(cleaned_copy, result)
        #cv2.imshow("Hybrid", line_result)
        
        key = cv2.waitKey(1)
            # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        

    rs.stop_capture()


    #terminate process
    sound_p.terminate()
    sound_p.join()