import numpy as np
import cv2
import sys
import time

# You should replace these 3 lines with the output in calibration step
DIM=(2592, 1944)
K=np.array([[1076.7148792467171, 0.0, 1298.9712963540678], [0.0, 1078.515014983842, 929.9968760065017], [0.0, 0.0, 1.0]])
D=np.array([[-0.016205134569390902], [-0.02434305021164351], [0.024555436941429715], [-0.008590717479362648]])


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