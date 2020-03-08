import tkinter as tk
import cv2
import darknet
import numpy as np
from skimage import io, draw
from skimage.util import img_as_float
import multiprocessing as mp
import time

###config###
thresh = 0.5
configPath = "./cfgs/yolov3_hr_c13.cfg"
weightPath = "./cfgs/yolov3_hr_c13_best.weights"
metaPath = "./cfgs/obj.data"
###config-end###

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def cam_check(cam):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    frame_count = 0
    if(cap is None or not cap.isOpened()):
        return False
    size = set_res(cap, 640,480)
    key = ''
    print("Camera:" + str(cam) + " Size:" + str(size))
    while(True):
        ret, frame = cap.read()
        cv2.imshow("Camera:" + str(cam) + " Press \'Y\' to Select or Press \'N\' to ignore", frame)
        key = cv2.waitKey(1) & 0xFF
        if(key == ord('Y') or key == ord('N')):
            break
    cap.release()
    cv2.destroyAllWindows()
    if(key == ord('Y')):
        return True
    return False

def cap_select():
    cam_left_num = -1
    cam_right_num = -1
    print("Select Camera Left(1-10):")
    for cam_num in range(1,10):
        cam_sel = cam_check(cam_num)
        if(cam_sel == True):
            cam_left_num = cam_num
            print("Select " + str(cam_left_num) + " As Camera Left(<<)")
            break
    if(cam_left_num == -1):
        print("Camera Left Not Available")

    print("Select Camera Right(1-10):")
    for cam_num in range(1,10):
        if(cam_num == cam_left_num):
            continue
        cam_sel = cam_check(cam_num)
        if(cam_sel == True):
            cam_right_num = cam_num
            print("Select " + str(cam_right_num) + " As Camera Right(>>)")
            break
    if(cam_right_num == -1):
        print("Camera Right Not Available")
    return cam_left_num, cam_right_num

def bgr2rgb_resized(img , network_width, network_height):
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
    return frame_resized

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    red = (255, 0, 0)
    green = (0, 255, 0)
    color = (0, 0, 0)
    for detection in detections:
        if("break" in detection[0]):
            color = red
        else:
            color = green
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0] +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
    return img

def cap_worker(cap_num, q, network_width, network_height, cap_diff, cap_speed):
    print("cap_worker START at cap:" + str(cap_num))
    count = 0
    cap = None
    #ifdemo
    if(isinstance(cap_num, str)):
        cap = cv2.VideoCapture(cap_num)
    else:
        cap = cv2.VideoCapture(cap_num, cv2.CAP_DSHOW)
    while(cap.isOpened()):
        ret, img = cap.read()
        if(ret):
            cv2.waitKey(round(cap_speed.value*cap_diff.value*1000))
            q.put(bgr2rgb_resized(img, network_width, network_height))


def do_detect(cap1_num, cap2_num):
    #load net
    darknet.performDetect(thresh=thresh, configPath=configPath, weightPath=weightPath, metaPath=metaPath, initOnly=True)
    network_width = darknet.network_width(darknet.netMain)
    network_height = darknet.network_height(darknet.netMain)
    print("network_width:" + str(network_width))
    print("network_height:" + str(network_height))

    #init windows
    cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Capture', network_width*2, network_height)

    #load a no_signal image
    no_signal_img = cv2.imread("./demo/no_signal.png")

    #Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(darknet.netMain), darknet.network_height(darknet.netMain),3)

    #init Queue and timediff
    imgq1 = mp.Queue()
    imgq2 = mp.Queue()
    cap_diff = mp.Value('d', 0.0)
    cap_speed = mp.Value('d', 1.5)

    #init cap_worker
    w1 = mp.Process(target=cap_worker,args=(cap1_num,imgq1,network_width,network_height,cap_diff,cap_speed,))
    w2 = mp.Process(target=cap_worker,args=(cap2_num,imgq2,network_width,network_height,cap_diff,cap_speed,))
    w1.start()
    w2.start()

    img1 = no_signal_img
    img2 = no_signal_img
    cv2.imshow("Capture", np.hstack((img1, img2)))

    while (True):
        FPS_count_start_time = time.time()
        Count_FPS = False
        if(imgq1.qsize() > 0 or imgq2.qsize() > 0):
            Count_FPS = True

        if(imgq1.qsize() > 0):
            detect_time_start = time.time()
            img1 = imgq1.get()
            darknet.copy_image_from_bytes(darknet_image, img1.tobytes())
            res1 = darknet.detect_image(net = darknet.netMain, meta = darknet.metaMain, im = darknet_image)
            img1 = cvDrawBoxes(res1, img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            cap_diff.value = round((time.time() - detect_time_start), 3)
            print("detect_time:" + str(cap_diff.value))

        if(imgq2.qsize() > 0):
            img2 = imgq2.get()
            darknet.copy_image_from_bytes(darknet_image, img2.tobytes())
            res2 = darknet.detect_image(net = darknet.netMain, meta = darknet.metaMain, im = darknet_image)
            img2 = cvDrawBoxes(res2, img2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img1 = cv2.resize(img1, dsize=(960,540))
        img2 = cv2.resize(img2, dsize=(960,540))
        
        if(Count_FPS):
            cv2.imshow("Capture", np.hstack((img1, img2)))
            print("FPS:" + str(round(60 / (time.time()-FPS_count_start_time), 1)))
            if(imgq1.qsize() > 2 or imgq2.qsize() > 2):
                cap_speed.value = cap_speed.value + 0.001
                print("decrease cap_speed:" + str(round(cap_speed.value, 3)))
        else:
            if(cap_speed.value > 0.001):
                cap_speed.value = cap_speed.value - 0.001
                print("increase cap_speed:" + str(round(cap_speed.value, 3)))
        print("imgq1.qsize():" + str(imgq1.qsize()) + " imgq2.qsize():" + str(imgq2.qsize()))
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break
        elif(key == ord('p')):
            key = cv2.waitKey(0)
    w1.terminate()
    w2.terminate()

if __name__ == '__main__':
    #demo mode switch
    demo = True
    cam_left_num = None
    cam_right_num = None
    if(demo is not True):
        cam_left_num, cam_right_num = cap_select()
    else:
        cam_left_num = "./demo/left.mp4"
        cam_right_num = "./demo/right.mp4"
    
    do_detect(cam_left_num, cam_right_num)