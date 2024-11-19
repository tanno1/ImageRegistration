# affine transformation function
import cv2
import numpy as np

def affine_transform(matches, unreg_read_image, kp_img, kp_ref):
    # for i in matches:
    #     print(f"match:{i.distance}")
    # matches first 3 values, NOT best 3 values
    if len(matches) >= 3:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        # print(src_pts)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        # print(dst_pts)

        # calulate affine matrix
        M = cv2.getAffineTransform(src_pts, dst_pts)
        # print(M)

        # affine img
        rows, cols, ch = unreg_read_image.shape
        img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))

    return img_aligned