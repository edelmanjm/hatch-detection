import time

import cv2
from grip import filterhatchpanel, filtervisiontarget
import numpy
import math
from muhthing import MuhThing
import processors
import os
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
# from pycallgraph import Config
import sys

w = 256
h = 144
framerate = 30

hatch_panel_pipeline = filterhatchpanel.GripPipeline()
vision_target_pipeline = filtervisiontarget.GripPipeline()

# degrees
angle = 23.5
inset = h * math.tan(math.radians(angle))
diagonal = h / math.cos(math.radians(angle))
# vertwarp = 1 / math.tan(math.radians(angle))
vertwarp = 1.1
warp = cv2.getPerspectiveTransform(
    # numpy.float32([[inset, 0], [w - inset, 0], [0, h], [w, h]]),
    # numpy.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
    numpy.float32([[0, 0], [w, 0], [-inset, h], [w + inset, h]]),
    numpy.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
)

K=numpy.array([[794.5616321293361, 0.0, 963.0391357869047], [0.0, 794.9001170024184, 498.968261322781], [0.0, 0.0, 1.0]])
D=numpy.array([[-0.019215744220979738], [-0.022168383678588813], [0.018999857407644722], [-0.003693599912847022]])

robot_mask = cv2.imread("./grip/robot_mask.png", cv2.IMREAD_REDUCED_GRAYSCALE_2)

# stream_url = "http://10.15.40.202:9001/cam.mjpg"
stream_url = ""


def find_hatches(source, draw=False):
    # Flatten the image
    img = cv2.warpPerspective(source, warp, (w, int(h * vertwarp)))

    # Detect the panels and find the centers
    contours = hatch_panel_pipeline.process(img)
    centers = processors.find_bounding_centers(contours)
    if draw:
        processors.draw_contours_and_centers(img, contours, centers)

    # Warp the image and back to the original
    img = cv2.warpPerspective(img, cv2.invert(warp)[1], (w, h))

    return img, contours, centers


def find_vision_target(source, draw=False):

    contours = vision_target_pipeline.process(source, robot_mask)
    centers = processors.find_bounding_centers(contours)

    # Find the two centers closed to the center of the image
    closest_distance = None
    closest_centers = [None, None]
    for center in centers:
        distance = math.sqrt((center[0] - w / 2)**2 + (center[1] - h / 2)**2)
        if closest_centers[1] is None or distance < closest_distance:
            closest_centers[1] = closest_centers[0]
            closest_centers[0] = center
            closest_distance = distance

    if closest_centers[1] is not None:
        if draw:
            processors.draw_contours_and_centers(source, contours, closest_centers)

        return source, contours, closest_centers
    else:
        return source, contours, []


def do_nothing(source, draw=False):
    return source, [], []


def main():

    # FIXME OpenCV refuses to read images with multiprocessing, maybe look into that

    print("Starting")

    thing = MuhThing(find_vision_target, "nothing", [w, h], camera_matrix=K, dist_coefficients=D, cam_stream=True, draw_contours=True)
    thing.start()


    print("Opening camera")
    if os.uname()[4] == 'armv7l':
        print("Using picamera")
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        # import time
        camera = PiCamera()
        camera.resolution = (w, h)
        camera.framerate = framerate
        camera.exposure_mode = 'off'
        camera.shutter_speed = 9000

        rawCapture = PiRGBArray(camera, size=(w, h))

        # allow the camera to warmup
        time.sleep(0.1)

        count = 0
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            thing.process_frame(frame.array)

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

            count += 1

            if count > 500:
                break
    else:
        print("Using cv2.VideoCapture")
        if stream_url == "":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        time.sleep(1)
        count = 0
        try:
            while count<200:
                _, raw = cap.read()
                thing.process_frame(raw)
                count += 1
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    # config = Config(max_depth=6)
    #
    # with PyCallGraph(output=GraphvizOutput()):
    #     main()

    main()
