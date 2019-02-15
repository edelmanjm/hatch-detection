import time

import cv2
from grip import filterhatchpanel, filtervisiontarget, filtervisiontarget2
import numpy
import math
from muhthing import MuhThing
import processors
import os
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
# from pycallgraph import Config

# Aspect ratio must be the same
w = 1440
h = 1080
w_low = 180
h_low = 135
w_max_process = w_low
h_max_process = h_low

framerate = 30
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
    # numpy.float32([[inset, 0], [w - inset, 0], [0, h], [w, h]]),
    # numpy.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
    numpy.float32([[0, 0], [w, 0], [-inset, h], [w + inset, h]]),
    numpy.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
)

scaled_K=numpy.array([[598.1749329148429, 0.0, 721.6507201967044], [0.0, 599.1750083243568, 516.6649311147231], [0.0, 0.0, 1.0]])
new_K=numpy.array([[231.16508956675534, 0.0, 724.6138722002406], [0.0, 231.55156935534703, 515.2072234471497], [0.0, 0.0, 1.0]])
D=numpy.array([[-0.019215744220979738], [-0.022168383678588813], [0.018999857407644722], [-0.003693599912847022]])

robot_mask = cv2.imread("./grip/robot_mask.png", cv2.IMREAD_REDUCED_GRAYSCALE_2)

stream_url = "http://10.15.40.202:9001/cam.mjpg"
# stream_url = ""


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
                                      interpolation=cv2.INTER_LINEAR)
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

        if draw:
            processors.draw_contours_and_centers(source, second_contours, second_centers)

        # return masked, second_contours, second_centers
        return source, second_contours, second_centers
    else:
        return source, [], []


def do_nothing(source, draw=False):
    return source, [], []


def main():

    # FIXME OpenCV refuses to read images with multiprocessing, maybe look into that

    print("Starting")

    thing = MuhThing(find_vision_target, "raspi-2", [w, h], scaled_K=scaled_K, new_K=new_K, dist_coefficients=D, cam_stream=True, draw_contours=True)
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
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # FIXME Temporary hack-y hack
            thing.process_frame(cv2.resize(frame.array, (w, h), interpolation=cv2.INTER_LINEAR))

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

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
