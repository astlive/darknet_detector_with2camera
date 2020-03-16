import cv2
import time
import os

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
        if(key == ord('Y')):
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif(key == ord('N')):
            cap.release()
            cv2.destroyAllWindows()
            return False
    cap.release()
    cv2.destroyAllWindows()
    return False

for nn in range(0,10):
    print(str(nn) + " " + str(cap_select(nn)))