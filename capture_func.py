import os
import cv2
import time

def sel_cap(cap_num):
    if(cap_num == 1):
        for i in range(10):
            cap = cv2.VideoCapture(i)
            

def test_cap(height,width,fps,cap_num,show_msg = False):
    if(show_msg):print()

if __name__ == "__main__":
    resolution = input("Select resolution (1)640x480 (2)1280x720 (3)1920x1080 (4)manual input:")
    if resolution == "1":
        x = 640
        y = 480
    elif resolution == "2":
        x = 1280
        y = 720
    elif resolution == "3":
        x = 1920
        y = 1080
    elif resolution == "4":
        x = int(input("width:"))
        y = int(input("height:"))
    
    cap_fps = input("Select FPS (1)30 (2)60 (3)120 (4)manual input:")
    if cap_fps == "1":
        fps = 30
    elif cap_fps == "2":
        fps = 60
    elif cap_fps == "3":
        fps = 120
    elif cap_fps == "4":
        fps = int(input("FPS:"))

    caps = input("Use how many capture:")
    if caps == "1":
        cap_num = 1
    elif caps == "2":
        cap_num = 2

    print("Test " + str(cap_num) + " Capture on " + str(x) + " x " + str(y) + " with FPS:" + str(fps))
    test_cap(x,y,fps,cap_num,True)