import cv2
import numpy as np

def display_corners_on_marker(img, corners):

    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    height, width = img.shape[:2]

    thickness = 2
    #Draw polylines
    points = np.array([[corners[0], corners[1]], [corners[2], corners[3]],
        [corners[6], corners[7]], [corners[4], corners[5]]])

    # Check corners limits are inside the image dimensions

    cv2.polylines(img_rgb, np.int32([points]), 1, (0,255,0), thickness)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_corners_on_marker(img, corners):

    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    height, width = img.shape[:2]

    thickness = 2
    #Draw polylines
    points = np.array([[corners[0], corners[1]], [corners[2], corners[3]],
        [corners[6], corners[7]], [corners[4], corners[5]]])

    # Check corners limits are inside the image dimensions

    cv2.polylines(img_rgb, np.int32([points]), 1, (0,255,0), thickness)
    return img_rgb

