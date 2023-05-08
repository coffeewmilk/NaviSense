import pyrealsense2 as rs
import numpy as np
import cv2
import pygame
import time
import threading
import multiprocessing as mp

def metronome(angle):

    pygame.mixer.init(frequency=44100, size=-16, channels=4)
    fifty_mp3 = pygame.mixer.Sound("./sounds/50BPM.mp3")
    letgo_mp3 = pygame.mixer.Sound("./sounds/ring.mp3")
    channel_close = pygame.mixer.Channel(0)
    channel_l = pygame.mixer.Channel(1)
    channel_r = pygame.mixer.Channel(2)
    channel_l_r = pygame.mixer.Channel(3)

    
    while True:
        print(f"angle feedback is {angle.value}")
        channel_close.set_volume(0,0)
        channel_l.set_volume(1,0)
        channel_r.set_volume(0,1)
        channel_l_r.set_volume(0.1,0.1)
        cons_angle = angle.value
        if cons_angle > 0:
            if cons_angle != 90:
                val = [ abs(90 - cons_angle), (90-cons_angle) ] 
                delay = 0.25 - ( (75-val[0])*(0.25-0.83) / (75-15) )
                print(delay)
                if val[1] > 0 :
                    channel_r.play(fifty_mp3,0)
                    print("R")
                    time.sleep(delay)
                    channel_r.stop()
                elif val[1] < 0 :
                    channel_l.play(fifty_mp3,0)
                    print("L")
                    time.sleep(delay)
                    channel_l.stop()
            else:
                delay = 1
                channel_l_r.play(letgo_mp3,0)
                print("M")
                time.sleep(delay)
                channel_l_r.stop()    
        
        
def distance_and_side (depth_img):

    output = []
    # find nearest distance
    left_depth = ("L", depth_img[0:427,236:473])
    mid_depth = ("M", depth_img[0:427,473:807])
    right_depth = ("R", depth_img[0:427,807:1044])
    for depth in [left_depth, mid_depth, right_depth]:
        for i in [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
            bg_removed_01_value = np.where((depth[1] > i) | (depth[1] <= 0),0,1)
            pixel =  np.sum(bg_removed_01_value) 
            density = (pixel/(427*334))*100
            if density :
                output.append( [depth[0],i] )
                break
        
    return output