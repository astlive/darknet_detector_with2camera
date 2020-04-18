import capture_func as cf
import multiprocessing as mp
import darknet
import time
import cv2
import numpy as np
import os

class Main:
    def __init__(self, darknetarg):
        self.darknetarg = darknetarg
        self.cap_ind1 = None
        self.cap_ind2 = None
        self.imgs = mp.Manager().Queue()
        self.imgds = mp.Manager().Queue()
        self.imgbs = mp.Manager().Queue()
        self.detector_ready = mp.Manager().Value('i', False)
        self.dn_width = mp.Manager().Value('i', 416)
        self.dn_height = mp.Manager().Value('i', 416)

    def roiDrawBoxes(self, detections, img, top = 0.1, bot = 0.1):
        #it's BGR in opencv
        red = (255, 0, 0)
        green = (0, 255, 0)
        color = (0, 0, 0)
        min_y = round(img.shape[0] * 0.1, 0)
        max_y = round(img.shape[0] * 0.9, 0)
        msg = ""

        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = cf.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)

            if("break" in detection[0] and ymin >= min_y and ymax <= max_y):
                color = red
                msg = "brk"
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
                cv2.putText(img,
                            detection[0] +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
            else:
                color = green
                #if there has other class do here.
        return msg, img

    def detector(self, skip = 2, debug = False):
        #for normal pc set skip = 2, Hight-End pc set skip = 1
        #7700hq+1060MaxQ 18fps on detector
        #2700x+2080ti 50fps on detector
        darknet.performDetect(thresh=self.darknetarg['thresh'], configPath=self.darknetarg['configPath'], 
                            weightPath=self.darknetarg['weightPath'], metaPath=self.darknetarg['metaPath'],
                            initOnly=True)
        self.dn_width.value = darknet.network_width(darknet.netMain)
        self.dn_height.value = darknet.network_height(darknet.netMain)
        darknet_image = darknet.make_image(self.dn_width.value, self.dn_height.value, 3)
        self.detector_ready.value = True
        skip_count = 0

        while True:
            if(self.imgs.empty()):
                time.sleep(1)
                if(debug):print("imgs queue is empty sleep 1s")
            else:
                fps_count_start_time = time.time()
                job = self.imgs.get(False)
                skip_count = skip_count + 1
                if(skip_count % skip == 0):
                    if(debug):print("job-->job[id] = " + str(job['id']) + " , job['img.shape']" + str(job['img'].shape))
                    darknet.copy_image_from_bytes(darknet_image, job['img'].tobytes())
                    detections = darknet.detect_image(net = darknet.netMain, meta = darknet.metaMain, im = darknet_image)
                    #move drawboxes to monitor() thread to improve performance(maybe?)
                    job['detections'] = detections
                    if(debug):print(detections)
                else:
                    if(debug):print("skip frame " + str(skip_count))
                    job['detections'] = []
                self.imgds.put(job)
                if(debug):print("detector FPS:" + str(round(1 / (time.time() - fps_count_start_time), 1)))

    def mergeframe(self, img1, img2, img3):
        frame = np.hstack((img1, img2))
        frame = np.hstack((frame, img3))
        return frame

    def monitor(self, debug = False):
        no_signal_img = cv2.resize(cv2.imread("./demo/no_signal.png"), dsize=(self.dn_width.value, self.dn_height.value))
        img1 = no_signal_img
        img2 = no_signal_img
        imgb = no_signal_img
        cv2.namedWindow("Monitor")
        frame = self.mergeframe(img1, img2, imgb)
        cv2.imshow("Monitor", frame)
        msgRead = True
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")

        while True:
            fps_count_start_time = time.time()
            u_frame = False
            if(not self.imgds.empty()):
                u_frame = True
                img = self.imgds.get(False)
                msg, img['img'] = self.roiDrawBoxes(img['detections'], img['img'])
                if(msg != ""):
                    time_str = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
                    img['time'] = time_str
                    img['img'] = cf.draw_msg(img['img'], time_str, img['id'])
                    self.imgbs.put(img)
                if(img['id'] == self.cap_ind1):img1 = img['img'][...,::-1]
                elif(img['id'] == self.cap_ind2):img2 = img['img'][...,::-1]
            if(not self.imgbs.empty() and msgRead):
                u_frame = True
                bimg = self.imgbs.get(False)
                imgb = bimg['img'][...,::-1]
                msgRead = False
            if(u_frame):
                frame = self.mergeframe(img1, img2, imgb)
                cv2.imshow("Monitor", frame)
                if(debug):print("imgs.qsize():" + str(self.imgs.qsize()) + " imgds.qsize():" + str(self.imgds.qsize())
                        + " imgbs.qsize():" + str(self.imgbs.qsize())
                        + " frame update FPS:" + str(round(1 / (time.time() - fps_count_start_time), 1)))
            key = cv2.waitKey(1)
            if(key == ord('q')):
                break
            elif(key == ord('p')):
                key = cv2.waitKey(0)
            elif(key == ord('Y') or key == ord('y')):
                if(not msgRead):
                    msgRead = True
                    f_path = os.path.join(save_path, bimg['time'] + ".jpg")
                    if(os.path.isfile(f_path)):
                        for i in range(100):
                            f_path = os.path.splitext(f_path)[0] + "-" + str(i) + ".jpg"
                            if(not os.path.isfile(f_path)):
                                break
                    cv2.imwrite(f_path, imgb)
                    if(self.imgbs.empty()):
                        imgb = no_signal_img
            elif(key == ord('N') or key == ord('n')):
                if(not msgRead):
                    msgRead = True
                    if(self.imgbs.empty()):
                        imgb = no_signal_img

    def run(self, demo = False, debug = True):
        mpdarknet = mp.Process(target=self.detector)
        mpdarknet.start()

        while self.detector_ready.value == False:
            print("Waiting for detector ready...re-check after 10s")
            time.sleep(10)

        if(demo):
            self.cap_ind1 = "./demo/gopro8(1).MP4"
            self.cap_ind2 = "./demo/gopro8(2).MP4"
        else:
            self.cap_ind1 = cf.sel_cap()
            if(debug):print("self.cap_ind1:" + str(self.cap_ind1))
            self.cap_ind2 = cf.sel_cap(skip = self.cap_ind1)
            if(debug):print("self.cap_ind2:" + str(self.cap_ind2))
        if(self.cap_ind1 is not None and self.cap_ind1 != -1):
            mpcap1 = mp.Process(target=cf.mp_cap_worker, args=(self.cap_ind1, self.dn_width,
                            self.dn_height, 60, self.imgs, "rgb",))
            mpcap1.start()
        if(self.cap_ind2 is not None and self.cap_ind2 != -1):
            mpcap2 = mp.Process(target=cf.mp_cap_worker, args=(self.cap_ind2, self.dn_width,
                            self.dn_height, 60, self.imgs, "rgb",))
            mpcap2.start()
        mpmon = mp.Process(target=self.monitor)
        mpmon.start()

        mpmon.join()
        mpdarknet.terminate()
        mpcap1.terminate()
        mpcap2.terminate()

if __name__ == "__main__":
    thresh = 0.5
    configPath = "./cfgs/model_1/csresnext50-panet-spp-original-optimal.cfg"
    weightPath = "./cfgs/model_1/csresnext50-panet-spp-original-optimal_best.weights"
    metaPath = "./cfgs/model_1/obj.data"
    darknetarg = {'thresh':thresh, 'configPath':configPath, 'weightPath':weightPath, 'metaPath':metaPath}
    main = Main(darknetarg = darknetarg)
    main.run()