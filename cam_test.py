import cv2 as cv
import numpy as np
import multiprocessing as mp
import time

def cam_init(source, size):
    cap = cv.VideoCapture(source)
    if(not size):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, size['x'])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, size['h'])
    return cap

def cam_worker(cap, cam_num, que):
    _ret, frame = cap.read()
    if(_ret):
        data = {'cam_num':cam_num, 'img':frame}


def main(cam_1, cam_2):
    qimg = mp.Manager().Queue()
    p = mp.Pool()
    cap1 = cam_init(cam_1)
    cap2 = cam_init(cam_2)
    p1 = p.apply_async(cam_worker,args=(cap1,1,qimg,))
    p2 = p.apply_async(cam_worker,args=(cap2,2,qimg,))
    no_signal_img = cv.imread("./demo/no_signal.png")
    no_signal_img = cv.resize(no_signal_img, dsize=(960,540))
    img_left = no_signal_img
    img_right = no_signal_img
    time.sleep(1)

    while(True):
        img = qimg.get()
        if(img['cam_num'] == 1):
            img_left = img
        elif(img['cam_num'] == 2):
            img_right = img
        cv.imshow("Capture", np.hstack((img_left, img_right)))
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break
        elif(key == ord('p')):
            key = cv2.waitKey(0)

if __name__ == '__main__':
    demo_left = "./demo/left.mp4"
    demo_right = "./demo/right.mp4"
    demo = True
    if(demo):
        main(demo_left, demo_right)