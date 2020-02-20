import tkinter as tk
import cv2

def envcheck():
    print("check the camera status")

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def cap_select(cam):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    if(cap is None or not cap.isOpened()):
        return False
    size = set_res(cap, 640,480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    key = ''
    print("Camera:" + str(cam) + " FPS:" + str(fps) + " Size:" + str(size))
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

def main():
    cam_left_num = -1
    cam_right_num = -1
    print("Select Camera Left(1-10):")
    for cam_num in range(1,10):
        cam_sel = cap_select(cam_num)
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
        cam_sel = cap_select(cam_num)
        if(cam_sel == True):
            cam_right_num = cam_num
            print("Select " + str(cam_right_num) + " As Camera Right(>>)")
            break
    if(cam_right_num == -1):
        print("Camera Right Not Available")
    


if __name__ == '__main__':
    main()