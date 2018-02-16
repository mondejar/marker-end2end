import cv2
import numpy as np

def display_corners_on_marker(img, corners):

    thickness = 2
    #Draw polylines
    points = np.array([[corners[0], corners[1]], [corners[2], corners[3]],
        [corners[6], corners[7]], [corners[4], corners[5]]])

    cv2.polylines(img, np.int32([points]), 1, (255), thickness)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()