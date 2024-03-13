import _init_paths
import NaviSense as ns
import open3d as o3d
import numpy as np
import cv2 
import sys
import multiprocessing
import time 
import PySimpleGUI as sg

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
    wrapped, _ = ns.getTransform(point2d, segmented)
    raw_wrapped, _ = ns.getTransform(point2d, raw_image)

    # cleanup some artifacts
    cleaned = ns.cleanup(wrapped)
    return cleaned, raw_wrapped

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

def transform(frame):
    #frame = np.array(frame)
    imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()  # can also use png.  ppm found to be more efficient
    return imgbytes

if __name__ == "__main__":

    #gui
    sg.theme('Dark')
    layout_l = [[sg.Text('Camera view', size=(15, 1), font='Helvetica 10')],
                [sg.Image(key='-IMAGE_cam-')],
                [sg.StatusBar(key='-STATUS-', text='angle')]]


    layout_r = [[sg.Image(key='-IMAGE_warp-')],
                [sg.Image(key='-IMAGE_path-')]]

    layout = [[sg.Col(layout_l), sg.Col(layout_r)]]
    window = sg.Window('Navisense 0.1', layout, no_titlebar=False, location=(0, 0), finalize=True)

    cam_elem = window['-IMAGE_cam-']
    warp_elem = window['-IMAGE_warp-']
    path_elem = window['-IMAGE_path-']
    status = window['-STATUS-']

    # initialize process2 
    queue_in = multiprocessing.Queue()
    queue_out = multiprocessing.Queue()
    process2p = multiprocessing.Process(target=process2, args=(queue_in, queue_out))
    process2p.start()


    # initialize feedback process and value
    angle = multiprocessing.Value('i', 90)
    sound_p = multiprocessing.Process(target=ns.metronome, args=(angle,))
    sound_p.start()

    

    while not bag_reader.is_eof():

        event, values = window.read(timeout=33)
        if event in ('Exit', None):
            break
        
        time1 = time.time()

        # put data in queue
        queue_in.put(im_rgbd)
        
        # perfprm process1
        img = cv2.cvtColor(np.asarray(im_rgbd.color), cv2.COLOR_BGR2RGB)
        cleaned, raw_wrap = process1(im_rgbd, cameraMatrix)
        # cv2.imshow('cleaned', cleaned)
        # cv2.imshow('rw', img)
        
        cam_elem.update(data=transform(img))
        warp_elem.update(data=transform(raw_wrap))

        # get value from the process here
        ob_points = queue_out.get()

        # overlay depthbased with segmentbased
        overlayed = ns.depthbased.obstacle_overlay(cleaned, ob_points)


        # calculate pathplan
        map = ns.create_occupacny_map(overlayed)
        result = ns.hybrid_maximum_angle(map, overlayed)
        
        # sent angle to feedback
        angle.value = ns.extract_angle(result)
        status.update(str(angle.value))

        # display the result
        line_result = ns.hybrid_drawLine(overlayed, result)
        #cv2.imshow("line", line_result)
        path_elem.update(data=transform(line_result))
        #print("this works")
        
        # key = cv2.waitKey(1)
        # if key == 27:
        #     cv2.destroyAllWindows()
        #     break
        im_rgbd = bag_reader.next_frame()

        time2 = time.time()
        #window.refresh()
        #sg.print(f'Time taken = {time2-time1}')

    bag_reader.close()

    #terminate process
    process2p.terminate()
    sound_p.terminate()
    process2p.join()
    sound_p.join()
