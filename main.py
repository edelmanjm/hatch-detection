import cv2
from grip import filterhatchpanel, filtervisiontarget
import numpy
import math
from muhthing import MuhThing
import processors

w = 1920
h = 1080

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

robot_mask = cv2.imread("./grip/robot_mask.png", cv2.IMREAD_REDUCED_GRAYSCALE_2)


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
    if draw:
        processors.draw_contours_and_centers(source, contours, centers)

    return source, contours, centers


if __name__ == "__main__":

    # MuhThing(find_hatches, "/hatch-centers", lambda: True).start()
    MuhThing(find_vision_target, "/target-centers", lambda: True).start()
