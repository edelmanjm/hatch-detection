import numpy
from networktables import NetworkTables
from mjpegserver import ThreadedHTTPServer, CamHandler
import threading
import time


class MuhThing:
    def __init__(self, contour_pipeline, name, dimensions, cam_stream=False, cam_stream_port=9001,
                 draw_contours=False):
        self.contour_pipeline = contour_pipeline
        self.name = name
        self.dimensions = dimensions
        self.cam_stream = cam_stream
        self.cam_stream_port = cam_stream_port
        self.draw_contours = draw_contours
        self.frame = numpy.zeros((dimensions[1], dimensions[0], 3), numpy.uint8)

    def process_frame(self, raw):
        start_time = time.time()
        processed, contours, centers = self.contour_pipeline(raw, self.draw_contours)
        self.frame = processed

        scaled_centers = []

        if len(centers) > 0:
            # Scale the centers into -1.0 to 1.0
            for center in centers:
                scaled_centers.append(
                    [center[0] / self.dimensions[0] * 2 - 1, center[1] / self.dimensions[1] * 2 - 1])

            # print(scaled_centers)
        else:
            # print("None")
            pass

        self.sd.putNumberArray(self.name + "/centers", [item for sublist in scaled_centers for item in sublist])
        print(time.time() - start_time)

    def start(self):

        print("Starting NetworkTables")
        NetworkTables.initialize(server='roborio-1540-frc.local')
        self.sd = NetworkTables.getTable(self.name)

        if self.cam_stream:
            print("Starting MJPEG stream")
            server = ThreadedHTTPServer(('', self.cam_stream_port), CamHandler, lambda *args: None, lambda *args: None,
                                        lambda: self.frame)
            threading.Thread(target=server.serve_forever).start()
