import cv2
import numpy
from networktables import NetworkTables
from mjpegserver import ThreadedHTTPServer, CamHandler
import threading


class MuhThing:
    def __init__(self, contour_pipeline, name, keep_alive, camera_port=0, width=1920, height=1080, cam_stream=False,
                 cam_stream_port=8080, draw_contours=False):
        self.contour_pipeline = contour_pipeline
        self.name = name
        self.keep_alive = keep_alive
        self.camera_port = camera_port
        self.width = width
        self.height = height
        self.cam_stream = cam_stream
        self.cam_stream_port = cam_stream_port,
        self.draw_contours = draw_contours

    def run(self, cap, sd):
        while self.keep_alive():
            _, raw = cap.read()
            processed, contours, centers = self.contour_pipeline(raw, self.draw_contours)
            if len(centers) > 0:
                print(centers)
            else:
                print("None")

            sd.putNumberArray(self.name + "/centers", [item for sublist in centers for item in sublist])

    def start(self):
        print("Starting")

        print("Opening camera")
        cap = cv2.VideoCapture(self.camera_port)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("Starting NetworkTables")
        NetworkTables.initialize(server='roborio-1540-frc.local')
        sd = NetworkTables.getTable(self.name)

        processed = None
        if self.cam_stream:
            print("Starting MJPEG stream")
            server = ThreadedHTTPServer(('127.0.0.1', self.cam_stream_port), CamHandler, lambda *args: None, lambda *args: None,
                                        lambda *args: processed if processed is not None else numpy.zeros(
                                            (self.width, self.height, 3), numpy.uint8))
            threading.Thread(target=server.serve_forever).start()

        # FIXME OpenCV refuses to read images with multiprocessing, maybe look into that
        print("Starting main process")
        threading.Thread(target=self.run, args=(cap, sd)).start()
