import tkinter as tk
import cv2

def envcheck():
    print("check the camera status")

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

def demo_cap():
    print("Try load from left.mp4 and right.mp4")

if __name__ == '__main__':
    demo = True
    cap_left, cap_right
    if(demo is not True):
        cam_left_num, cam_right_num = cap_select()
        cap_left = cv2.VideoCapture(cam_left_num)
        cap_right = cv2.VideoCapture(cam_right_num)
    else:
        cap_left = cv2.VideoCapture("./demo/left.mp4")
        cap_right = cv2.VideoCapture("./demo/right.mp4")
    set_res(cap_left, 1920, 1080)
    set_res(cap_right, 1920, 1080)
    