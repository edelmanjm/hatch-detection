import numpy as np
import cv2
import sys
import time

# You should replace these 3 lines with the output in calibration step
DIM=(1920, 1080)
K=np.array([[794.5616321293361, 0.0, 963.0391357869047], [0.0, 794.9001170024184, 498.968261322781], [0.0, 0.0, 1.0]])
D=np.array([[-0.019215744220979738], [-0.022168383678588813], [0.018999857407644722], [-0.003693599912847022]])


def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    start_time = time.time()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("./" + sys.argv[1] + "_undistorted.jpg", undistorted_img)
    print("Took " + str(time.time() - start_time))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)