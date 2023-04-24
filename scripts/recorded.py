import NaviSense
import open3d as o3d
import numpy as np
import cv2 
import sys

# path for recorded bag file
recorded_file = ""

# initialize bag reader and config
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(recorded_file)
im_rgbd = bag_reader.next_frame()
width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
intrinsic_t = o3d.core.Tensor(cameraMatrix)

# initialize segmentation model
model = seg.init_model(small=True)



while not bag_reader.is_eof():
    point2d = getPlane(im_rgbd, cameraMatrix)
    test = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)
    test = demo.segmentationfun(model, test)
    wrapped, masked = getTransform(point2d, test)
    cv2.imshow("Mask", masked)
    cv2.imshow("wrappped", wrapped)
    cleaned = cleanup(wrapped)
    cv2.imshow("Cleanuped", cleaned)
    value = sp.max_value_angle(sp.create_occupacny_map(cleaned))
    print(value)
    origin_x = int(wrapped.shape[0]/2)
    origin_y = int(wrapped.shape[1])
    origin = (origin_x, origin_y)
    points = sp.angle_to_point(origin, value)
    cleaned_line = cv2.line(cleaned, (origin_x, origin_y), points, [255,0,0], 2)
    cv2.imshow("line", cleaned_line)
   
    key = cv2.waitKey(1)
        # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
    im_rgbd = bag_reader.next_frame()

bag_reader.close()