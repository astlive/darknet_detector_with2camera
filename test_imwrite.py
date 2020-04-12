import os
import cv2
import time
time_str = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
img = cv2.imread("./demo/dog.jpg")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves", time_str + ".jpg")
print(path)
cv2.imwrite(path, img)