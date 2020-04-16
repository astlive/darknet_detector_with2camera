import os
import cv2
import time
import numpy as np

def cv_size(img):
    return tuple(img.shape[1::-1])

def draw_msg(img, str1, str2, color1 = (0,255,0), color2 = (0,0,255)):
    cv2.putText(img, str1, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color1, 2)
    cv2.putText(img, str2, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color2, 2)
    return img

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def init_cap(cap_index, width, height, fps, debug = True):
    if(debug):print("init_cap args-->cap_index:" + str(cap_index) + " width:" + str(width) + " height:"
                + str(height) + " fps:" + str(fps))
    if(not isinstance(width, int)):width = width.value
    if(not isinstance(height, int)):height = height.value
    if(not isinstance(fps, int)):fps = fps.value
    cap = None
    cap = cv2.VideoCapture(cap_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print("init VideoCapture on width:" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + " height:" + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), " fps:" + str(cap.get(cv2.CAP_PROP_FPS)))
    return cap

def sel_cap(skip = -1):
    for i in range(10):
        cap = init_cap(i, 640, 480, 60)
        if(cap is None or not cap.isOpened() or i == skip):
            continue
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "cv2.CAP_PROP_FPS:" + str(cap.get(cv2.CAP_PROP_FPS)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.imshow("Camera:" + str(i), frame)
                key = cv2.waitKey(1)
                if(key == ord('Y') or key == ord('y')):
                    return i
                elif(key == ord('N') or key == ord('n')):
                    break
        cap.release()
        cv2.destroyAllWindows()
    return None

def mp_cap_worker(cap_ind, width, height, fps, q, mode="bgr"):
    import cv2
    if(not isinstance(width, int)):width = width.value
    if(not isinstance(height, int)):height = height.value
    if(not isinstance(fps, int)):fps = fps.value
    cap = init_cap(cap_ind, width, height, fps)
    print("mp_cap_worker Process:" + str(os.getpid()) + " cap_ind:" + str(cap_ind))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            if(cv_size(frame) != (width,height)):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if(mode == "rgb"):frame = frame[...,::-1]
            #[...,::-1] switch b and r at image (numpy operation [start:end:step])
            r = {'id':cap_ind,'img':frame}
            q.put(r)
            cv2.waitKey(10)

def test_cap(width,height,fps,cap_num,demo = False):
    if(demo is False):
        cap_ind1 = sel_cap()
    else:
        cap_ind1 = "./demo/gopro8(1).MP4"

    if(cap_num == 1):
        cap_1 = init_cap(cap_ind1, width, height, fps)
        while(cap_1.isOpened()):
            ret, frame = cap_1.read()
            if(ret):
                if(cv_size(frame) != (width,height)):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(frame, "Camera:" + str(cap_ind1), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.imshow("Capture", frame)
                key = cv2.waitKey(1)
                if(key == ord('Q') or key == ord('q')):
                    break
        cap_1.release()
    elif(cap_num == 2):
        import multiprocessing as mp
        if(demo is False):
            cap_ind2 = sel_cap(cap_ind1)
        else:
            cap_ind2 = "./demo/gopro8(2).MP4"

        imgq1 = mp.Manager().Queue()
        imgq2 = mp.Manager().Queue()
        w1 = mp.Process(target=mp_cap_worker, args=(cap_ind1,width,height,fps,imgq1,))
        w2 = mp.Process(target=mp_cap_worker, args=(cap_ind2,width,height,fps,imgq2,))
        w1.start()
        w2.start()

        no_signal_img = cv2.imread("./demo/no_signal.png")
        cv2.resize(no_signal_img, dsize=(width,height))
        img1 = no_signal_img
        img2 = no_signal_img
        
        while (True):
            FPS_count_start_time = time.time()
            count_fps = False
            if(imgq1.qsize() > 0 and imgq2.qsize() > 0):
                count_fps = True
            
            show_frame = False
            if(imgq1.qsize() > 0 or imgq2.qsize() > 0):
                show_frame = True
            

            if(imgq1.qsize() > 0):
                detect_time_start = time.time()
                img1 = imgq1.get()['img']
            else:
                cv2.putText(img1, "X", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

            if(imgq2.qsize() > 0):
                img2 = imgq2.get()['img']
            else:
                cv2.putText(img2, "X", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
            
            if(show_frame):
                if(img1.shape != img2.shape):
                    img1 = cv2.resize(img1, dsize=(width,height))
                    img2 = cv2.resize(img2, dsize=(width,height))
                frame = np.hstack((img1, img2))
                if(demo):print("img1.shape:" + str(img1.shape) + " img2.shape:" + str(img2.shape))
                if(demo):print("frame.shape:" + str(frame.shape))
                cv2.imshow("Capture", frame)
                if(count_fps):print("FPS:" + str(round(1 / (time.time() - FPS_count_start_time), 1)))
            if(demo):print("imgq1.qsize():" + str(imgq1.qsize()) + " imgq2.qsize():" + str(imgq2.qsize()))
            key = cv2.waitKey(1)
            if(key == ord('q')):
                break
            elif(key == ord('p')):
                key = cv2.waitKey(0)
        w1.terminate()
        w2.terminate()


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
    test_cap(x,y,fps,cap_num,demo = True)