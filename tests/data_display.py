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
recorded_file = "../Video/extra/20230504_171902.bag"
#recorded_file = "../Video/extra/20230504_171902.bag"

# initialize bag reader and config
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(recorded_file)
im_rgbd = bag_reader.next_frame()
width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
intrinsic_t = o3d.core.Tensor(cameraMatrix)

# initialize segmentation model
model = ns.seg.init_model()

# start at this time
#bag_reader.seek_timestamp(53024000)
#bag_reader.seek_timestamp(33870000)
l1 = 300
l2 = 500

while not bag_reader.is_eof():

    #segmented and cleaned
    cleaned = process1(im_rgbd, cameraMatrix)

    #cv2.imshow("Mask", masked)
    #cv2.imshow("wrappped", wrapped)
    cv2.imshow("Cleanuped", cleaned)

    
    cleaned_copy = np.copy(cleaned)
    map = ns.create_occupacny_map(cleaned)
    angle, value = ns.max_value_angle(map, l1)
    angle2, value2 = ns.max_value_angle(map, l2)
    #print(angle)

    C_value, O_value  = ns.obstacle_value(angle, cleaned, l1)
    C_value2, O_value2 = ns.obstacle_value(angle2, cleaned, l2)

    
    ns.drawLine(cleaned, angle2, l2)
    ns.drawLine(cleaned, angle, l1, [0,255,0])
    cv2.putText(img=cleaned, text=f'Certaity value: {str(C_value)}', org=(0, 300), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),thickness=1)
    cv2.putText(img=cleaned, text=f'Obstacle value: {str(O_value)}', org=(0, 310), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0),thickness=1)
    cv2.putText(img=cleaned, text=f'Certaity value: {str(C_value2)}', org=(0, 320), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),thickness=1)
    cv2.putText(img=cleaned, text=f'Obstacle value: {str(O_value2)}', org=(0, 330), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0),thickness=1)
    cv2.putText(img=cleaned, text=f'value1: {str(value)}', org=(0, 350), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),thickness=1)
    cv2.putText(img=cleaned, text=f'value2: {str(value2)}', org=(0, 370), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),thickness=1)
    cv2.imshow("line", cleaned)
   
    #test new algorithm
    result = ns.hybrid_maximum_angle(map, cleaned_copy)
    line_result = ns.hybrid_drawLine(cleaned_copy, result)
    cv2.imshow("Hybrid", line_result)
    

    key = cv2.waitKey(1)
        # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
    im_rgbd = bag_reader.next_frame()

bag_reader.close()