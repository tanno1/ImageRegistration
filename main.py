# image registration main file
# noah tanner, jianya wei
# fall 2024

import cv2
import numpy as np
from harris2 import harris_corner
from match import match
import sys

# detailed usage:
# input:
#
# output:


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
    cv2.imshow('unreg', unreg_read_image)
    cv2.imshow('ref', ref_read_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # match corners
    matched_images = match(unreg_gray_image, ref_gray_image, unreg_read_image, ref_read_image)

    cv2.imshow('matched corners', matched_images)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])