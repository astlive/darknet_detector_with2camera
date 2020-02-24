import tkinter as tk
import cv2
import darknet
import numpy as np
from skimage import io, draw
from skimage.util import img_as_float

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

def draw_boxes(img, detections):
    print("*** "+str(len(detections))+" Results, color coded by confidence ***")
    imcaption = []
    for detection in detections:
        label = detection[0]
        confidence = detection[1]
        pstring = label+": "+str(np.rint(100 * confidence))+"%"
        imcaption.append(pstring)
        print(pstring)
        bounds = detection[2]
        shape = img.shape
        # x = shape[1]
        # xExtent = int(x * bounds[2] / 100)
        # y = shape[0]
        # yExtent = int(y * bounds[3] / 100)
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        # Coordinates are around the center
        xCoord = int(bounds[0] - bounds[2]/2)
        yCoord = int(bounds[1] - bounds[3]/2)
        boundingBox = [
            [xCoord, yCoord],
            [xCoord, yCoord + yExtent],
            [xCoord + xEntent, yCoord + yExtent],
            [xCoord + xEntent, yCoord]
        ]
        # Wiggle it around to make a 3px border
        rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
        rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
        rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
        rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
        rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
        boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
        draw.set_color(img, (rr, cc), boxColor, alpha= 0.8)
        draw.set_color(img, (rr2, cc2), boxColor, alpha= 0.8)
        draw.set_color(img, (rr3, cc3), boxColor, alpha= 0.8)
        draw.set_color(img, (rr4, cc4), boxColor, alpha= 0.8)
        draw.set_color(img, (rr5, cc5), boxColor, alpha= 0.8)
    detections = {
        "detections": detections,
        "image": img,
        "caption": "\n<br/>".join(imcaption)
    }
    return detections

def bgr2rgb_resized(img):
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(darknet.netMain),
                                darknet.network_height(darknet.netMain)),
                                interpolation=cv2.INTER_LINEAR)
    return frame_resized

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
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
                    [0, 255, 0], 2)
    return img

def do_detect(cap1, cap2):
    #config
    thresh = 0.5
    configPath = "./cfgs/yolov3_hr_c13.cfg"
    weightPath = "./cfgs/yolov3_hr_c13_best.weights"
    metaPath= "./cfgs/obj.data"
    destroyWindowCount = 0

    #load net
    darknet.performDetect(thresh=thresh, configPath=configPath, weightPath=weightPath, metaPath=metaPath, initOnly=True)

    #init windows
    cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Capture', 1920, 1080)

    #load a no_signal image
    no_signal_img = cv2.imread("./demo/no_signal.png")

    #Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(darknet.netMain), darknet.network_height(darknet.netMain),3)

    while (cap1.isOpened() or cap2.isOpened()):
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        if not ret1:
            img1 = no_signal_img
        if not ret2:
            img2 = no_signal_img

        if (not ret1) and (not ret2):
            print("Both 2 Capture no signal..." + str(destroyWindowCount))
            destroyWindowCount = destroyWindowCount + 1
            if (destroyWindowCount == 60):
                cv2.destroyWindow('Capture')
                cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Capture', 1920, 1080)
                destroyWindowCount = 0
                print("Reset window")
        else:
            destroyWindowCount = 0

            img1_resized = bgr2rgb_resized(img1)
            darknet.copy_image_from_bytes(darknet_image, img1_resized.tobytes())
            res1 = darknet.detect_image(net = darknet.netMain, meta = darknet.metaMain, im = darknet_image)
            img1 = cvDrawBoxes(res1, img1_resized)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            img2_resized = bgr2rgb_resized(img2)
            darknet.copy_image_from_bytes(darknet_image, img2_resized.tobytes())
            res2 = darknet.detect_image(net = darknet.netMain, meta = darknet.metaMain, im = darknet_image)
            img2 = cvDrawBoxes(res2, img2_resized)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            img1 = cv2.resize(img1, dsize=(960,540))
            img2 = cv2.resize(img2, dsize=(960,540))
        
        cv2.imshow('Capture', np.hstack((img1, img2)))
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break
        elif(key == ord('p')):
            key = cv2.waitKey(0)
    cap1.release()
    cap2.release()

if __name__ == '__main__':
    #demo mode switch
    demo = True
    cap_left = None
    cap_right = None
    if(demo is not True):
        cam_left_num, cam_right_num = cap_select()
        cap_left = cv2.VideoCapture(cam_left_num)
        cap_right = cv2.VideoCapture(cam_right_num)
    else:
        cap_left = cv2.VideoCapture("./demo/left.mp4")
        cap_right = cv2.VideoCapture("./demo/right.mp4")
    set_res(cap_left, 1280, 720)
    set_res(cap_right, 1280, 720)
    
    do_detect(cap_left, cap_right)