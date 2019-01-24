import cv2
import grip
import numpy
import math

w = 640
h = 360

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
cap.set(cv2.CAP_PROP_EXPOSURE, 5)

grip_pipeline = grip.GripPipeline()

# degrees
angle = 23.5
inset = h * math.tan(math.radians(angle))
diagonal = h / math.cos(math.radians(angle))
# vertwarp = 1 / math.tan(math.radians(angle))
vertwarp = 1.2
warp = cv2.getPerspectiveTransform(
    # numpy.float32([[inset, 0], [w - inset, 0], [0, h], [w, h]]),
    # numpy.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
    numpy.float32([[0, 0], [w, 0], [-inset, h], [w + inset, h]]),
    numpy.float32([[0, 0], [w, 0], [0, h * vertwarp], [w, h * vertwarp]])
)

while True:
    _, raw = cap.read()

    img = cv2.warpPerspective(raw, warp, (w, int(h * vertwarp)))
    contours = grip_pipeline.process(img)
    for contour in contours:
        cv2.moments(contour)
        br_x, br_y, br_w, br_h = cv2.boundingRect(contour)
        cv2.drawMarker(img, (int(br_x + br_w / 2), int(br_y + br_h / 2)), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # cv2.imshow('my webcam', img)
    cv2.imshow('my webcam', cv2.warpPerspective(img, cv2.invert(warp)[1], (w, h)))

    if cv2.waitKey(1) == 27:
        break  # esc to quit


cv2.destroyAllWindows()