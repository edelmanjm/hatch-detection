import time

import cv2
from grip import filterhatchpanel, filtervisiontarget, filtervisiontarget2
import numpy as np
import math
from muhthing import MuhThing
import processors
import os
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
# from pycallgraph import Config

# Aspect ratio must be the same
w = 640
h = 480
w_low = 90
h_low = 68
w_max_process = w_low
h_max_process = h_low

framerate = 90
crop_margin = 10

hatch_panel_pipeline = filterhatchpanel.GripPipeline()
vision_target_pipeline_1 = filtervisiontarget.GripPipeline(w_low, h_low)
vision_target_pipeline_2 = filtervisiontarget2.GripPipeline()

# degrees
angle = 23.5
inset = h * math.tan(math.radians(angle))
diagonal = h / math.cos(math.radians(angle))
# vertwarp = 1 / math.tan(math.radians(angle))
vertwarp = 1.1
warp = cv2.getPerspectiveTransform(
    # np.float32([[inset, 0], [w - inset, 0], [0, h], [w, h]]),
    # np.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
    np.float32([[0, 0], [w, 0], [-inset, h], [w + inset, h]]),
    np.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
)

scaled_K=np.array([[337.96669256003423, 0.0, 298.6639042576305], [0.0, 338.49353082497515, 235.04426655713644], [0.0, 0.0, 1.0]])
new_K=np.array([[163.96201251537047, 0.0, 283.0155708637292], [0.0, 164.21760415824954, 235.094308364941], [0.0, 0.0, 1.0]])
D=np.array([[-0.11260320941729775], [0.054187476530898164], [-0.04172171303039251], [0.011310942683823726]])

robot_mask = cv2.imread("./grip/robot_mask.png", cv2.IMREAD_REDUCED_GRAYSCALE_2)

stream_url = "http://10.20.150.6:1181/stream.mjpg"
# stream_url = "http://10.20.150.6:9001/cam.mjpg"
# stream_url = ""


def find_hatches(source):
    # Flatten the image
    img = cv2.warpPerspective(source, warp, (w, int(h * vertwarp)))

    # Detect the panels and find the centers
    contours = hatch_panel_pipeline.process(img)
    centers = processors.find_bounding_centers(contours)

    # Warp the image and back to the original
    img = cv2.warpPerspective(img, cv2.invert(warp)[1], (w, h))

    return img, contours, centers


def find_vision_target_simple(source):
    # TODO apply robot mask
    first_contours = vision_target_pipeline_2.process(source, None)

    first_bounding_rects = processors.find_bounding_rects(first_contours)
    first_centers = processors.find_bounding_centers(first_bounding_rects)

    # Find the two centers closed to the center of the image
    closest_distance = None
    closest_bounding_rects = [None, None]
    for i, center in enumerate(first_centers):
        distance = math.sqrt((center[0] - w_low / 2)**2 + (center[1] - h_low / 2)**2)
        if closest_bounding_rects[1] is None or distance < closest_distance:
            closest_bounding_rects[1] = closest_bounding_rects[0]
            closest_bounding_rects[0] = first_bounding_rects[i]
            closest_distance = distance

    if closest_bounding_rects[1] is not None:

        # return masked, second_contours, second_centers
        return source, first_contours, first_centers
    else:
        return source, [], []


def find_vision_target(source):

    # TODO apply robot mask
    first_contours = vision_target_pipeline_1.process(source)

    first_bounding_rects = processors.find_bounding_rects(first_contours)
    first_centers = processors.find_bounding_centers(first_bounding_rects)

    # Find the two centers closed to the center of the image
    closest_distance = None
    closest_bounding_rects = [None, None]
    for i, center in enumerate(first_centers):
        distance = math.sqrt((center[0] - w_low / 2)**2 + (center[1] - h_low / 2)**2)
        if closest_bounding_rects[1] is None or distance < closest_distance:
            closest_bounding_rects[1] = closest_bounding_rects[0]
            closest_bounding_rects[0] = first_bounding_rects[i]
            closest_distance = distance

    if closest_bounding_rects[1] is not None:

        # TODO remove because we'll be better
        # If the closest ones are too close to the edge, just don't bother
        if closest_distance > math.sqrt((w * (3/4)) ** 2 + (h * (3/4)) ** 2):
            return source, [], []

        x0 = min(closest_bounding_rects[0][0], closest_bounding_rects[1][0]) / w_low * w - crop_margin
        x1 = max(closest_bounding_rects[0][0] + closest_bounding_rects[0][2], closest_bounding_rects[1][0] + closest_bounding_rects[1][2]) / w_low * w + crop_margin
        y0 = min(closest_bounding_rects[0][1], closest_bounding_rects[1][1]) / h_low * h - crop_margin
        y1 = max(closest_bounding_rects[0][1] + closest_bounding_rects[0][3], closest_bounding_rects[1][1] + closest_bounding_rects[1][3]) / h_low * h + crop_margin

        x0 = int(round(x0))
        x1 = int(round(x1))
        y0 = int(round(y0))
        y1 = int(round(y1))

        if x0 < 0:
            x0 = 0
        if x1 > w:
            x1 = w
        if y0 < 0:
            y0 = 0
        if y1 > h:
            y1 = h

        # TODO mask of parts around the targets
        masked = source[y0:y1, x0:x1]
        max_process_limit_factor = 1
        if y1 - y0 > h_max_process or x1 - x0 > w_max_process:
            max_process_limit_factor = min(w_max_process / (x1 - x0), h_max_process / (y1 - y0))
            second_image = cv2.resize(masked,
                                      (int(round((x1 - x0) * max_process_limit_factor)),
                                       int(round((y1 - y0) * max_process_limit_factor))),
                                      interpolation=cv2.INTER_NEAREST)
        else:
            second_image = masked
        second_contours = vision_target_pipeline_2.process(second_image, None)
        second_centers = processors.find_bounding_centers(processors.find_bounding_rects(second_contours))

        # Transform centers and contours so they're relative to the source image instead of the copped image
        for contour in second_contours:
            for point in contour:
                point[0][0] = point[0][0] * (1 / max_process_limit_factor) + x0
                point[0][1] = point[0][1] * (1 / max_process_limit_factor) + y0
        for center in second_centers:
            center[0] = center[0] * (1 / max_process_limit_factor) + x0
            center[1] = center[1] * (1 / max_process_limit_factor) + y0

        # return masked, second_contours, second_centers
        return source, second_contours, second_centers
    else:
        return source, [], []


def do_nothing(source, draw=False):
    return source, [], []


def main():

    # FIXME OpenCV refuses to read images with multiprocessing, maybe look into that

    print("Starting")

    thing = MuhThing(find_vision_target, "raspi-2", (w, h), scaled_K=scaled_K, new_K=new_K, dist_coefficients=D, cam_stream=True, draw_contours=True)
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
        camera.shutter_speed = 7000
        camera.awb_mode = 'off'
        camera.awb_gains = (1.2, 1.9)

        rawCapture = PiRGBArray(camera, size=(w, h))

        # allow the camera to warmup
        time.sleep(0.1)

        count = 0
        # capture frames from the camera
        last_time = time.time()
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            thing.process_frame(frame.array)

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            current_time = time.time()
            print("Loop time: " + str(round((current_time - last_time) * 1000, 3)) + "ms")
            last_time = current_time


            # count += 1

            # if count > 50:
            #     break
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
            while count < 250:
                _, raw = cap.read()
                thing.process_frame(raw)
                count += 1
        except KeyboardInterrupt:
            pass
        # thing.process_frame(cv2.imread("/Users/Jonathan/Desktop/big.jpg"))


if __name__ == "__main__":
    # config = Config(max_depth=6)
    #
    # with PyCallGraph(output=GraphvizOutput()):
    #     main()

    main()
