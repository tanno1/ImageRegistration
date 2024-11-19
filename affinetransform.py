# affine transformation function
import cv2
import numpy as np

def affine_transform(matches, unreg_read_image, kp_img, kp_ref):
    # for i in matches:
    #     print(f"match:{i.distance}")
    # matches 5
    if len(matches) >= 5:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in matches[:5]]).reshape(-1, 1, 2)
        # print(src_pts)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches[:5]]).reshape(-1, 1, 2)
        # print(dst_pts)

        # calulate affine matrix
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        # print(M)

        # affine img
        rows, cols, ch = unreg_read_image.shape
        img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))

    else:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        # print(src_pts)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        # print(dst_pts)
        print(f"LOW POINT MATCHING FOR IMAGE: {unreg_read_image}")

        # calulate affine matrix
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        # print(M)

        # affine img
        rows, cols, ch = unreg_read_image.shape
        img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))

    return img_aligned