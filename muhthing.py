import numpy as np
from networktables import NetworkTables
from mjpegserver import ThreadedHTTPServer, CamHandler
import threading
import time
import cv2


class MuhThing:
    def __init__(self, contour_pipeline, name, dimensions, scaled_K=None, new_K=None, dist_coefficients=None, cam_stream=False,
                 cam_stream_port=9001, draw_contours=False):
        self.contour_pipeline = contour_pipeline
        self.name = name
        self.dimensions = dimensions
        self.scaled_K = scaled_K
        self.new_K = new_K
        self.dist_coefficients = dist_coefficients
        self.cam_stream = cam_stream
        self.cam_stream_port = cam_stream_port
        self.draw_contours = draw_contours
        self.frame = np.zeros((dimensions[1], dimensions[0], 3), np.uint8)

        # dimensions_undistorted = np.zeros((1, 2, 2), dtype=np.float32)
        # original_dimensions = np.array([[[0, 0], [dimensions[0], dimensions[1]]]], dtype=np.float32)
        # cv2.undistortPoints(original_dimensions,
        #                     self.camera_matrix, self.dist_coefficients, dimensions_undistorted)
        # self.dimensions_undistorted = [
        #     dimensions_undistorted[0][1][0] - dimensions_undistorted[0][0][0],
        #     dimensions_undistorted[0][1][1] - dimensions_undistorted[0][0][1]
        # ]

    def process_frame(self, raw):
        start_time = time.time()
        processed, contours, centers = self.contour_pipeline(raw, self.draw_contours)
        self.frame = processed

        vectors = np.empty((2, 3), dtype=np.float64)
        angles = np.empty((2, 1), dtype=np.float64)

        if len(centers) == 2:
            # Remap to 1xN 2-channel
            # Undistort the points
            if self.scaled_K is not None and self.dist_coefficients is not None:
                for i, center in enumerate(centers):
                    vectors[i] = np.matmul(
                        np.linalg.inv(
                            self.scaled_K
                        ),
                        # np.array([320, 240, 1])
                        np.append(
                            cv2.fisheye.undistortPoints(
                                # np.array([[[100, 100]]], dtype=np.float64),
                                np.array([[center]], dtype=np.float64),
                                self.scaled_K,
                                self.dist_coefficients,
                                R=np.eye(3),
                                P=self.new_K
                            )[0][0],
                            1
                        ),
                    )

            for i, vector in enumerate(vectors):
                angles[i] = np.rad2deg(np.arctan(vector[0] / vector[2]))
                # angles[i] = np.rad2deg(np.arccos(
                #     np.clip(np.dot(vector, np.array([0, 0, 1], dtype=np.float64)) / (np.linalg.norm(vector)), -1, 1)))
        else:
            # print("None")
            pass

        # self.sd.putNumberArray("vectors", [item for sublist in vectors for item in sublist])
        final_value = []
        if vectors is not None:
            final_value = vectors.flatten()
        self.sd.putNumberArray("vectors", final_value)
        NetworkTables.flush()

        print("Found " + str(centers.__len__()) + " targets, latency " + str(round((time.time() - start_time) * 1000)) + "ms: " + str(angles.flatten()))

    def start(self):

        print("Starting NetworkTables")
        NetworkTables.initialize(server='roborio-1540-frc.local')
        self.sd = NetworkTables.getTable(self.name)

        if self.cam_stream:
            print("Starting MJPEG stream")
            server = ThreadedHTTPServer(('', self.cam_stream_port), CamHandler, lambda *args: None, lambda *args: None,
                                        lambda: self.frame)
            threading.Thread(target=server.serve_forever).start()
