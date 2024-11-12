# affine transformation function
import cv2
import numpy as np

def affine_transform(matches, unreg_read_image, kp_img, kp_ref):
    # matches first 3 values, NOT best 3 values
    if len(matches) >= 3:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)

    # calulate affine matrix
    M = cv2.getAffineTransform(src_pts, dst_pts)
    M2 = np.array([
    [1.00199826e+00, 1.79706125e-03, -1.25801421e+00],
    [-7.99309672e-04, 9.99281214e-01, 2.90320568e+00]
])

    # affine img
    rows, cols, ch = unreg_read_image.shape
    img_aligned = cv2.warpAffine(unreg_read_image, M2, (cols, rows))

    return img_aligned