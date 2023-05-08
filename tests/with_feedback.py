import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys
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
    return (cleaned, masked)




if __name__ == "__main__":

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

    # initialize feedback process and value
    angle = multiprocessing.Value('i', 90)
    sound_p = multiprocessing.Process(target=ns.metronome, args=(angle,))
    sound_p.start()


    while not bag_reader.is_eof():

        #segmented and cleaned
        cleaned, masked = process1(im_rgbd, cameraMatrix)
        cv2.imshow("Cleanuped", cleaned)
        cv2.imshow("masked", masked)


        map = ns.create_occupacny_map(cleaned)
        angle.value, value = ns.max_value_angle(map)
        #print(angle)


        cleaned_line = ns.drawLine(cleaned, angle.value)
        cv2.imshow("line", cleaned_line)
    
        key = cv2.waitKey(1)
            # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        im_rgbd = bag_reader.next_frame()

    bag_reader.close()

    #terminate process
    sound_p.terminate()
    sound_p.join()