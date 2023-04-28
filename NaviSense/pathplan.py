import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2



walkway = np.array([244, 35, 232])
obstacle = np.array([128, 64, 128])

def create_occupacny_map(img):
    occupancy_map = np.ones(img.shape[:2], dtype=np.int64)
    occupancy_map &= np.all(img == walkway, axis=2)
    dist_transform = distance_transform_edt(occupancy_map)
    return dist_transform


def angle_to_point(origin, angle):
    angle_rad = np.deg2rad(angle)
    origin_x = origin[0]
    origin_y = origin[1]
    l = 300
    x = origin_x + l*np.cos(angle_rad)
    y = origin_y - l*np.sin(angle_rad)
    return (int(x), int(y))


def mask_from_angle(shape, angle):
    mask = np.zeros(shape[:2])
    origin_x = int(shape[0]/2)
    origin_y = int(shape[1])
    points = angle_to_point((origin_x, origin_y), angle)
    thickness = 10
    mask = cv2.line(mask, (origin_x, origin_y), points, 1, thickness).astype(bool)
    return mask

def value_of_angle(distant_transformed, angle):
    mask = mask_from_angle(distant_transformed.shape, angle)
    mask = mask.astype(bool)
    #print(dist_transform)
    return np.sum(distant_transformed[mask])

def obstacle_value(angle, map):
    mask = mask_from_angle(map.shape, angle)
    covered = map[mask]
    C_value = np.sum(np.all(covered==walkway, axis=1))
    O_value = np.sum(np.all(covered==obstacle, axis=1))

    return (C_value, O_value)



    

def max_value_angle(distant_transformed):
    angles = np.array([15,30,45,60,75,90,105,120,135,150,165,180])
    values = np.array([value_of_angle(distant_transformed,i) for i in angles])
    index = np.argmax(values)
    maxangle = angles[index]
    maxvalue = values[index]
    return (maxangle, maxvalue)

def drawLine(cleaned, value):
    origin_x = int(cleaned.shape[0]/2)
    origin_y = int(cleaned.shape[1])
    origin = (origin_x, origin_y)
    points = angle_to_point(origin, value)
    color = [255,0,0]
    weight = 2
    cleaned_line = cv2.line(cleaned, (origin_x, origin_y), points, color, weight)
    return cleaned_line
