import cv2


def find_bounding_rects(contours):
    return [cv2.boundingRect(contour) for contour in contours]


def find_bounding_centers(bounding_rects):
    centers = []
    for br_x, br_y, br_w, br_h in bounding_rects:
        center_x = br_x + br_w / 2
        center_y = br_y + br_h / 2
        centers.append([center_x, center_y])
    return centers


def draw_contours_and_centers(img, contours, centers):
    for center in centers:
        cv2.drawMarker(img, (int(center[0]), int(center[1])), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)