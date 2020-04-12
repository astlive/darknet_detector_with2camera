import darknet
import numpy as np
import multiprocessing as mp
import time
import os

###config###
thresh = 0.5
configPath = "./cfgs/model_1/csresnext50-panet-spp-original-optimal.cfg"
weightPath = "./cfgs/model_1/csresnext50-panet-spp-original-optimal_best.weights"
metaPath = "./cfgs/model_1/obj.data"
fps_skip = 1
###config-end###

def set_res(cap, x,y):
    import cv2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def cam_check(cam):
    import cv2
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    frame_count = 0
    if(cap is None or not cap.isOpened()):
        return False
    size = set_res(cap, 1280,720)
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
    import cv2
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
    import cv2
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
    import cv2
    red = (0, 0, 255)
    green = (0, 255, 0)
    color = (0, 0, 0)
    min_y = round(img.shape[0] * 0.1, 0)
    max_y = round(img.shape[0] * 0.9, 0)
    send_dialog = False
    msg = ""
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if("break" in detection[0] and ymin > min_y and ymax < max_y):
            color = red
            send_dialog = True
            msg = "brk"
        else:
            color = green
        
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0] +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
    return msg, img

def show_dialog(q):
    import cv2
    # rr = {'msg':msg, 'src':str(cap1_num), 'img':img1}
    cfm_list = list()
    first_run = True
    index = 0
    while(True):
        if(q.qsize() > 0):
            # cfm_list.append(q.get())
            rr = q.get()
            time_str = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
            cv2.putText(rr['img'], time_str, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.imshow("list", rr['img'])
            key = cv2.waitKey(0)
            if(key == ord('Y') or key == ord('y')):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves", time_str + ".jpg")
                if(os.path.isfile(path)):
                    for i in range(100):
                        path = os.path.splitext(path)[0] + "-" + str(i) + ".jpg"
                        if(not os.path.isfile(path)):
                            break
                cv2.imwrite(path, rr['o_img'])
            elif(key == ord('N') or key == ord('n')):
                print("N")

    #     if(len(cfm_list) > 0):
    #         img = cfm_list[0]['img']
    #         cv2.putText(img, "如果確認，按Y，否則N", (10,20),cv2.FONT_HERSHEY_COMPLEX,12,(0,0,255),5)
    #         cv2.imshow("list", cfm_list[0]['img'])
    # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    
    # tit_str = str(cam_num) + "Capture Break "
    # if(msg == "brk"):
    #     cv2.putText(frame, "如果確認，按Y，否則N", (10,20),cv2.FONT_HERSHEY_COMPLEX,12,(0,0,255),5)
    #     cv2.imshow(tit_str, frame)
    #     key = cv2.waitKey(0)
    #     if(key == ord('Y') or key == ord('y')):
    #         print("Y")
    #     elif(key == ord('N') or key == ord('n')):
    #         print("N")
    return 0

def cap_worker(cap_num, q, network_width, network_height, cap_diff, cap_speed):
    import cv2
    print("cap_worker START at cap:" + str(cap_num))
    count = 0
    cap = None
    #ifdemo
    if(isinstance(cap_num, str)):
        cap = cv2.VideoCapture(cap_num)
    else:
        cap = cv2.VideoCapture(cap_num)
        set_res(cap, 1280,720)
        cap.set(cv2.CAP_PROP_FPS, 60)
    while(cap.isOpened()):
        pre_time = time.time()
        ret, img = cap.read()
        if(ret):
            cvt_time = time.time()
            q.put(bgr2rgb_resized(img, network_width, network_height))
            print("camera time taken:" + str(round(time.time() - pre_time, 2)) + "s")
            print("cvt time taken:" + str(round(time.time() - cvt_time, 2)) + "s")
            cv2.waitKey(round(cap_speed.value*cap_diff.value*1000))

def do_detect(cap1_num, cap2_num):
    #load net
    darknet.performDetect(thresh=thresh, configPath=configPath, weightPath=weightPath, metaPath=metaPath, initOnly=True)
    network_width = darknet.network_width(darknet.netMain)
    network_height = darknet.network_height(darknet.netMain)
    #Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(darknet.netMain), darknet.network_height(darknet.netMain),3)

    #init Queue and timediff
    imgq1 = mp.Manager().Queue()
    imgq2 = mp.Manager().Queue()
    imgqb = mp.Manager().Queue()
    cap_diff = mp.Value('d', 0.0)
    cap_speed = mp.Value('d', 1.5)

    #init cap_worker
    print("network_height:" + str(network_height) + " network_width:" + str(network_width))
    w1 = mp.Process(target=cap_worker,args=(cap1_num,imgq1,network_width,network_height,cap_diff,cap_speed,))
    w2 = mp.Process(target=cap_worker,args=(cap2_num,imgq2,network_width,network_height,cap_diff,cap_speed,))
    w3 = mp.Process(target=show_dialog,args=(imgqb,))
    w1.start()
    w2.start()
    w3.start()

    #init windows
    import cv2
    cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Capture', network_width*2, network_height)
    #load a no_signal image
    no_signal_img = cv2.imread("./demo/no_signal.png")
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
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            o_img = img1.copy()
            msg, img1 = cvDrawBoxes(res1, img1)
            if(msg != ""):
                rr = {'msg':msg, 'src':str(cap1_num), 'img':img1, 'o_img':o_img}
                imgqb.put(rr)
            cap_diff.value = round((time.time() - detect_time_start), 3)
            print("detect_time:" + str(cap_diff.value))

        if(imgq2.qsize() > 0):
            img2 = imgq2.get()
            darknet.copy_image_from_bytes(darknet_image, img2.tobytes())
            res2 = darknet.detect_image(net = darknet.netMain, meta = darknet.metaMain, im = darknet_image)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            o_img = img2.copy()
            msg, img2 = cvDrawBoxes(res2, img2)
            if(msg != ""):
                rr = {'msg':msg, 'src':str(cap2_num), 'img':img2, 'o_img':o_img}
                imgqb.put(rr)
        
        if(Count_FPS):
            dis_rr_time = time.time()
            if(img1.shape != img2.shape):
                img1 = cv2.resize(img1, dsize=(960,540))
                img2 = cv2.resize(img2, dsize=(960,540))
            cv2.imshow("Capture", np.hstack((img1, img2)))
            print("FPS:" + str(round(1 / (time.time()-FPS_count_start_time), 1)))
            print("display result time taken:" + str(round(time.time() - dis_rr_time, 2)) + "s")
            if(imgq1.qsize() > 2 or imgq2.qsize() > 2):
                cap_speed.value = cap_speed.value + 0.001
                print("decrease cap_speed:" + str(round(cap_speed.value, 3)))
        else:
            if(cap_speed.value > 1):
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
    w3.terminate()

if __name__ == '__main__':
    #demo mode switch
    demo = True
    cam_left_num = None
    cam_right_num = None
    if(demo is not True):
        cam_left_num, cam_right_num = cap_select()
    else:
        cam_left_num = "./demo/gopro8(1).MP4"
        cam_right_num = "./demo/gopro8(2).MP4"
    
    do_detect(cam_left_num, cam_right_num)