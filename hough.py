import cv2
import numpy as np

# read
def hough_transform(img_gray):
    # parameters
    threshold1 = 50
    threshold2 = 150
    apertureSize = 3
    edges = cv2.Canny(img_gray, threshold1, threshold2, apertureSize)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    return lines


def draw_lines(lines, img):
    cv2.imshow(img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 + 1000 * (-b))
            y2 = int(y0 + 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# if __name__ == "__main__":
#     hough_transform()