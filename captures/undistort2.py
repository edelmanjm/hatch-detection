import numpy as np
import cv2
import sys


# You should replace these 3 lines with the output in calibration step
DIM=(2592, 1944)
K=np.array([[1076.7148792467171, 0.0, 1298.9712963540678], [0.0, 1078.515014983842, 929.9968760065017], [0.0, 0.0, 1.0]])
D=np.array([[-0.016205134569390902], [-0.02434305021164351], [0.024555436941429715], [-0.008590717479362648]])


def undistort(img_path, balance=1.0, dim2=(1440, 1080), dim3=(1440, 1080)):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite("./" + sys.argv[1] + "_undistorted.jpg", undistorted_img)
    print("scaled_K=np.array(" + str(scaled_K.tolist()) + ")")
    print("new_K=np.array(" + str(new_K.tolist()) + ")")
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)