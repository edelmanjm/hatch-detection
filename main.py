import cv2
import grip
import numpy
import math
from networktables import NetworkTables
from mjpegserver import ThreadedHTTPServer, CamHandler
import threading

w = 1280
h = 720

grip_pipeline = grip.GripPipeline()

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


def find_hatches(source, draw=False):
    # Flatten the image
    img = cv2.warpPerspective(source, warp, (w, int(h * vertwarp)))

    # Detect the panels and find the centers
    contours = grip_pipeline.process(img)
    centers = []
    for contour in contours:
        cv2.moments(contour)
        br_x, br_y, br_w, br_h = cv2.boundingRect(contour)
        center_x = br_x + br_w / 2
        center_y = br_y + br_h / 2
        centers.append([center_x, center_y])
        if draw:
            cv2.drawMarker(img, (int(center_x), int(center_y)), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
    if draw:
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # Warp the image and back to the original
    img = cv2.warpPerspective(img, cv2.invert(warp)[1], (w, h))

    return img, contours, centers


if __name__ == "__main__":

    print("Starting")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    cap.set(cv2.CAP_PROP_EXPOSURE, 5)

    print("Starting NetworkTables")
    NetworkTables.initialize(server='roborio-1540-frc.local')
    sd = NetworkTables.getTable("hatch-cam")

    run = True
    processed = None
    print("Starting MJPEG stream")
    server = ThreadedHTTPServer(('127.0.0.1', 8080), CamHandler, lambda *args: None, lambda *args: None, lambda *args: processed if processed is not None else numpy.zeros((h, w, 3), numpy.uint8))
    threading.Thread(target=server.serve_forever).start()

    while run:
        _, raw = cap.read()
        processed, contours, centers = find_hatches(raw, True)
        if len(centers) > 0:
            print(centers)
        else:
            print("None")
        # cv2.imshow('my webcam', processed)
        if cv2.waitKey(1) == 27:
            run = False  # esc to quit

        sd.putNumberArray("hatch-centers", [item for sublist in centers for item in sublist])

    cv2.destroyAllWindows()