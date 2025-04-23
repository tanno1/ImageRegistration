# image registration
# match corners

import cv2
import numpy as np

def match(img_gray, ref_gray, unreg_read_img, ref_read_img):
    # Debugging: Check input image properties
    print(f"img_gray shape: {img_gray.shape}, dtype: {img_gray.dtype}")
    print(f"ref_gray shape: {ref_gray.shape}, dtype: {ref_gray.dtype}")
    print(f"unreg_read_img dtype: {unreg_read_img.dtype}")
    print(f"ref_read_img dtype: {ref_read_img.dtype}")

    # Convert float32 images to uint8
    if img_gray.dtype == np.float32:
        img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ref_gray.dtype == np.float32:
        ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Normalize 16-bit images to 8-bit range (if applicable)
    if img_gray.dtype == np.uint16:
        img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ref_gray.dtype == np.uint16:
        ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Normalize 16-bit input images for drawMatches
    if unreg_read_img.dtype == np.uint16:
        unreg_read_img = cv2.normalize(unreg_read_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ref_read_img.dtype == np.uint16:
        ref_read_img = cv2.normalize(ref_read_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ORB feature detection
    orb = cv2.ORB_create()
    kp_img, desc_img = orb.detectAndCompute(img_gray, None)
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_img, desc_ref)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw match result
    match_image = cv2.drawMatches(unreg_read_img, kp_img, ref_read_img, kp_ref, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return match_image, kp_img, kp_ref, matches

if __name__ == '__main__':
    match()