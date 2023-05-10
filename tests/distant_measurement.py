import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys
import json
import time



def process1(im_rgbd, cameraMatrix):
    global model
    
    # detect floor plane
    point2d = ns.getPlane(im_rgbd, cameraMatrix)

    # image segmentation
    raw_image = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)

    # transfrom to top view
    wrapped, masked = ns.getTransform(point2d, raw_image)

    return raw_image, wrapped

def mouse_callback(event, x, y, flags, frame):
    if event == 2:
        print(f"coords {x, y}")
        frame_copy = np.copy(frame)
        cv2.putText(img=frame_copy, text=f'Coordinates: x = {x}, y = {y}', org=(100, 400), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),thickness=1)
        cv2.circle(frame_copy, (x,y), 30, (0, 255, 0), 1)
        cv2.imwrite(f"./tests/saved/{time.time()}.png", frame_copy)


if __name__ == "__main__":
    # camera parameters
    width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
    cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    intrinsic_t = o3d.core.Tensor(cameraMatrix)

    # initialize segmentation model
    model = ns.seg.init_model()

    # initialize and config camera
    with open('./Scripts/rs_config.json') as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(False)  # true: start recording with capture

    while(True):
        
        # capture data and align them
        im_rgbd = rs.capture_frame(True, True)  
        raw, wrapped = process1(im_rgbd, cameraMatrix)
        cv2.imshow("Just wrap", wrapped)
        cv2.setMouseCallback("Just wrap", mouse_callback, wrapped)
        cv2.imshow("raw", raw)

        
        key = cv2.waitKey(1)
            # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        

    rs.stop_capture()

