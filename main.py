# image registration main file
# noah tanner, jianya wei
# fall 2024

import cv2
import numpy as np
from harris2 import harris_corner
from match import match
from hough import hough_transform, draw_lines
from affinetransform import affine_transform
import sys

def main(unreg_image, ref_image):
    # open and read each image
    unreg = cv2.imread(unreg_image)
    ref = cv2.imread(ref_image)
    # convert images to grayscale
    gray_unreg = cv2.cvtColor(unreg, cv2.COLOR_BGR2GRAY)
    ref_unreg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # read images
    unreg_read_image = cv2.imread(unreg_image)
    ref_read_image = cv2.imread(ref_image)


    # convert to grayscale
    unreg_gray_image = cv2.cvtColor(unreg_read_image, cv2.COLOR_BGR2GRAY)
    unreg_gray_mat = np.float32(unreg_gray_image)
    ref_gray_image = cv2.cvtColor(ref_read_image, cv2.COLOR_BGR2GRAY)
    ref_gray_mat = np.float32(ref_gray_image)

    # apply harris corner calculations
    unreg_dst = harris_corner(unreg_gray_mat)
    unreg_read_image[unreg_dst > 0.05 * unreg_dst.max()] = [0, 0, 255]
    ref_dst = harris_corner(ref_gray_mat)
    ref_read_image[ref_dst > 0.05 * ref_dst.max()] = [0, 0, 255]
    cv2.imshow('Harris Corner Unregistered', unreg_read_image)
    cv2.imshow('Harris Corner Reference', ref_read_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # match corners
    matched_images, kp_img, kp_ref, matches = match(unreg_gray_image, ref_gray_image, unreg_read_image, ref_read_image)

    cv2.imshow('Corner Matching (Unregistered Left, Reference Right)', matched_images)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # affine transformation
    img_aligned = affine_transform(matches, unreg_read_image, kp_img, kp_ref)

    cv2.imshow('ALigned Image', img_aligned)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # error calcualtion
    reg_gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    print(type(reg_gray))
    print(type(ref_unreg))
    error = cv2.absdiff(reg_gray, ref_unreg)
    error_enhanced = cv2.normalize(error, None, 0, 255, cv2.NORM_MINMAX)
    mean_error = np.mean(error)
    print(f'Mean Error: {mean_error}')
    cv2.imshow('Error Image', error)
    cv2.imshow('Enhanced Error Image', error_enhanced)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # # hough transform (implement later)
    # hough_lines_img = hough_transform(ref_read_image)
    # hough_lines_ref = hough_transform(ref_read_image)
    # print(hough_lines_img)
    # print(hough_lines_ref)

    # # draw hough lines on imgs
    # draw_lines(hough_lines_img, unreg_read_image)
    # draw_lines(hough_lines_ref, ref_read_image)

    # cv2.imshow('Hough lines unregistered img', unreg_read_image)
    # cv2.imshow('Hough lines reference img', ref_read_image)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])