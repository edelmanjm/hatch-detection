import numpy
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
        self.frame = numpy.zeros((dimensions[1], dimensions[0], 3), numpy.uint8)

        # dimensions_undistorted = numpy.zeros((1, 2, 2), dtype=numpy.float32)
        # original_dimensions = numpy.array([[[0, 0], [dimensions[0], dimensions[1]]]], dtype=numpy.float32)
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

        # scaled_centers = []
        undistorted_centers = None

        if len(centers) > 0:
            # Remap to 1xN 2-channel
            undistorted_centers = numpy.asarray([centers], dtype=numpy.float32)
            # Undistort the points
            if self.scaled_K is not None and self.dist_coefficients is not None:
                # FIXME appears to not be normalizing the points now
                cv2.undistortPoints(undistorted_centers, self.scaled_K, self.dist_coefficients, undistorted_centers, numpy.eye(3), self.new_K)

            # FIXME remove
            # Scale the centers into -1.0 to 1.0
            for center in undistorted_centers[0]:
                center[0] = center[0] / 1440 * 2 - 1
                center[1] = center[1] / 1080 * 2 - 1
        else:
            # print("None")
            pass

        # self.sd.putNumberArray(self.name + "/centers", [item for sublist in scaled_centers for item in sublist])
        final_value = []
        if undistorted_centers is not None:
            final_value = undistorted_centers.flatten()
        self.sd.putNumberArray("centers", final_value)
        NetworkTables.flush()

        print("Found " + str(centers.__len__()) + " targets, latency " + str(round((time.time() - start_time) * 1000)) + "ms: " + str(final_value))

    def start(self):

        print("Starting NetworkTables")
        NetworkTables.initialize(server='roborio-1540-frc.local')
        self.sd = NetworkTables.getTable(self.name)

        if self.cam_stream:
            print("Starting MJPEG stream")
            server = ThreadedHTTPServer(('', self.cam_stream_port), CamHandler, lambda *args: None, lambda *args: None,
                                        lambda: self.frame)
            threading.Thread(target=server.serve_forever).start()
