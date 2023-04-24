import cv2 
import numpy as np
import time

#this is really not a good idea but will fix it later!!
color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (  0,  0,  0)]
black = (0,0,0)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

def smoothandfill(img):    
    dilation = cv2.dilate(img, dilate_kernel, iterations=1)
    erosion = cv2.erode(dilation, erode_kernel, iterations=1)
    smooth = cv2.medianBlur(erosion,9)
    #output = np.where(smooth != black, nclass, black)
    return smooth

def color_similar(color):
    color = np.array(color)
    similarlity = [np.sum((np.array(each) - color)**2) for each in color_map]
    return (color_map[np.argmin(similarlity)])


def color_similar_optimized(color):
    color = np.array(color)
    global color_map
    color_map = np.array(color_map)
    color_diff = color_map - color
    similarlity = np.sum(color_diff**2, axis=1)
    return (color_map[np.argmin(similarlity)])



def cleanup(img):
    scaled = cv2.convertScaleAbs(img, alpha=(255.0))
    clean = smoothandfill(scaled)
    mask = np.ones(img.shape[:2], dtype=bool)
    for color in color_map:
        mask &= ~np.all(clean == color, axis=2)
    temp = np.apply_along_axis(color_similar_optimized,1,clean[mask])
    clean[mask] = temp
    return clean