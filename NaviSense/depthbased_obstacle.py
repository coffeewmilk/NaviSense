import _init_paths
import numpy as np
import cv2
import open3d as o3d
from matplotlib import pyplot as plt
import time
from transformation import rearrange


def upperbody_obstacle_crop(rgbd, instrinstic):
    pc = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, instrinstic, depth_max = 5.0)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    downpcl = pc.voxel_down_sample(voxel_size=0.04)
    
    # create selection box
    points = o3d.core.Tensor([[-3, -0.3, -1.5], [3, -0.3, -1.5], [3, -0.3, -5], [3, 0.6, -5]])
    box = o3d.t.geometry.AxisAlignedBoundingBox.create_from_points(points)

    #crop point cloud
    downpcl_crop = downpcl.crop(box)

    return downpcl_crop 

def extractRegion(pc):
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels = pc.cluster_dbscan(eps=0.25, min_points=30, print_progress=False)
    max_label = labels.max().item()

    boxes = []
    if max_label != 0:
        for label in (range(max_label + 1)):
            obs = pc.select_by_mask(labels==label)
            obs_box = obs.get_axis_aligned_bounding_box()
            color = o3d.core.Tensor([1.0,  0.0, 0.0], o3d.core.float32)
            obs_box.set_color(color)
            obs_box = obs_box.to_legacy()
            center = obs_box.get_center()
            extent = obs_box.get_half_extent()
            points = obs_box.get_box_points()
            boxes.append((center, extent, points))
    return boxes

def show_region(img, regions):
    img_copy = np.copy(img)
    width = 1280; height = 720; fx = 649.302; fy = fx; cx = 649.934; cy = 335.776
    cameraMatrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    for region in regions:
        x3d, y3d, z3d = region[0]
        z3d += region[1][2]
        coor = cv2.projectPoints((x3d, y3d, z3d),(3.14,0,0),(0,0,0),cameraMatrix,0)[0].reshape(1,2)
        x, y = coor[0].astype(int)
        cv2.putText(img_copy, text=f'{-z3d}m', org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0),thickness=1)
        cv2.circle(img_copy, (x,y), 50, (0, 255, 0), 1)
    return img_copy

def extract_points(boxes):
    points2d = []
    for box in boxes:
        points = np.asarray(box[2])
        points = points[points[:,1].argsort()][:4]
        x3d = points[:,0]
        z3d = points[:,2]
        x = x3d*72 + 262
        y = (z3d+3)*72 +393
        print(f'z3d = {z3d}, y = {y}')
        points2d.append((np.stack((x,y), axis=1)).astype(int))
    return points2d

def obstacle_overlay(map, points2d):
    map_copy = np.copy(map)
    for each in points2d:
        cv2.drawContours(map_copy, [rearrange(each)], -1, (255, 255, 255), -1)
    return map_copy