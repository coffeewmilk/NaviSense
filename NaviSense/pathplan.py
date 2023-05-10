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


def angle_to_point(origin, angle, l=300):
    angle_rad = np.deg2rad(angle)
    origin_x = origin[0]
    origin_y = origin[1]
    x = origin_x + l*np.cos(angle_rad)
    y = origin_y - l*np.sin(angle_rad)
    return (int(x), int(y))


def mask_from_angle(shape, angle, l=300):
    mask = np.zeros(shape[:2])
    origin_x = int(shape[0]/2)
    origin_y = int(shape[1])
    points = angle_to_point((origin_x, origin_y), angle, l)
    thickness = 10
    mask = cv2.line(mask, (origin_x, origin_y), points, 1, thickness).astype(bool)
    return mask

def value_of_angle(distant_transformed, angle, l=300):
    mask = mask_from_angle(distant_transformed.shape, angle, l)
    #print(dist_transform)
    return np.sum(distant_transformed[mask])

def obstacle_value(angle, map, l=300):
    mask = mask_from_angle(map.shape, angle, l)
    covered = map[mask]
    W_value = np.sum(np.all(covered==walkway, axis=1))
    O_value = np.sum(np.all(covered==obstacle, axis=1))
    return (W_value, O_value)
  
def path_value(distant_transformed, map, angle, l):
    mask = mask_from_angle(map.shape, angle, l)
    mask_obstacle = mask_from_angle(map.shape, angle, l-200)
    distance_value = np.sum(distant_transformed[mask])
    covered = map[mask]
    covered_obstacle = map[mask_obstacle]
    W_value = np.sum(np.all(covered==walkway, axis=1))
    O_value = np.sum(np.all(covered_obstacle==obstacle, axis=1))
    return (distance_value, W_value, O_value)


def max_value_angle(distant_transformed, l=300):
    angles = np.array([15,30,45,60,75,90,105,120,135,150,165,180])
    values = np.array([value_of_angle(distant_transformed,i,l) for i in angles])
    index = np.argmax(values)
    maxangle = angles[index]
    maxvalue = values[index]
    return (maxangle, maxvalue)

def drawLine(cleaned, value, l=300, color=[255,0,0]):
    origin_x = int(cleaned.shape[0]/2)
    origin_y = int(cleaned.shape[1])
    origin = (origin_x, origin_y)
    points = angle_to_point(origin, value, l)
    weight = 2
    cleaned_line = cv2.line(cleaned, (origin_x, origin_y), points, color, weight)
    return cleaned_line

def hybrid_maximum_angle(distant_transformed, map):
    l1 = 300
    l2 = 500
    angles = [15,30,45,60,75,90,105,120,135,150,165,180]
    values1 = []
    values2 = []
    max_values1 = -1
    max_values2 = -1
    for angle in angles:
        d_value1, W_value1, O_value1 = path_value(distant_transformed, map, angle, l1)
        d_value2, W_value2, O_value2 = path_value(distant_transformed, map, angle, l2)
        if ((O_value1 < 50) and (d_value1 > 8000) and (W_value1 > 800) and (max_values1 < d_value1)):
            max_values1 = d_value1
            values1 = [angle, max_values1, W_value1, O_value1]
        if ((O_value1 < 50) and (max_values2 < d_value2) and (W_value2 > W_value1)):
            max_values2 = d_value2
            values2 = [angle, max_values2, W_value2, O_value2]
    return (values1, values2)


def extract_angle(data):
    if ((len(data[0]) != 0) or (len(data[1]) != 0)):
        if len(data[0]) != 0:
            return data[0][0]
        return data[1][0]
    return -1





def hybrid_drawLine(cleaned, data):
    origin_x = int(cleaned.shape[0]/2)
    origin_y = int(cleaned.shape[1])
    origin = (origin_x, origin_y)
    weight = 2
    color1 = [255,0,0]
    color2 = [0,255,0]
    
    if ((len(data[0]) != 0) or (len(data[1]) != 0)):
        if len(data[0]) != 0:
            points_s = angle_to_point(origin, data[0][0], 100)
            points = angle_to_point(origin, data[0][0], 300)
            line = cv2.line(cleaned, (origin_x, origin_y), points, color1, weight)
            line = cv2.line(cleaned, (origin_x, origin_y), points_s, color2, weight)
            return line
        points_s = angle_to_point(origin, data[1][0], 300)
        points_l = angle_to_point(origin, data[1][0], 500)
        line = cv2.line(cleaned, (origin_x, origin_y), points_l, color1, weight)
        line = cv2.line(line, (origin_x, origin_y), points_s, color2, weight)
        return line
    else:
        error = cv2.putText(img=cleaned, text='No path found', org=(0, 250), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255),thickness=1)
        return error


            




