# image registration main file
# noah tanner, jianya wei
# fall 2024

import cv2
import numpy as np
from harris2 import harris_corner
from match import match
from affinetransform import affine_transform
import sys

def main(unreg_image, ref_image, sp, show):
    # enable or disable subpixel, 1 = yes, 0 = no subpixel
    print(show)
    subpixel = sp

    # open and read each image
    unreg = cv2.imread(unreg_image)
    ref = cv2.imread(ref_image)

    # convert images to grayscale
    gray_unreg = cv2.cvtColor(unreg, cv2.COLOR_BGR2GRAY)
    ref_unreg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # read images
    unreg_read_image = cv2.imread(unreg_image)
    ref_read_image = cv2.imread(ref_image)

    unreg_read_image_2 = cv2.imread(unreg_image)
    ref_read_image_2 = cv2.imread(ref_image)

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
    if show == '1':
        cv2.imshow('Harris Corner Unregistered Image', unreg_read_image)
        cv2.imshow('Harris Corner Reference Image', ref_read_image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # subpixel accuracy
    ## comment out until line 69 if no subpixel
    if subpixel == 1:
        img_sp = cv2.imread(unreg_image)
        gray_sp = cv2.cvtColor(img_sp,cv2.COLOR_BGR2GRAY)

        # find Harris corners
        gray_sp = np.float32(gray_sp)
        dst_sp = cv2.cornerHarris(gray_sp,5,3,0.04)
        dst_sp = cv2.dilate(dst_sp,None)
        ret_sp, dst_sp = cv2.threshold(dst_sp,0.01*dst_sp.max(),255,0)
        dst_sp = np.uint8(dst_sp)

        # find centroids
        ret_sp, labels_sp, stats_sp, centroids_sp = cv2.connectedComponentsWithStats(dst_sp)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray_sp,np.float32(centroids_sp),(5,5),(-1,-1),criteria)

        # Now draw them
        res_sp = np.hstack((centroids_sp,corners))
        res_sp = np.intp(res_sp)
        img_sp[res_sp[:,1],res_sp[:,0]]=[0,0,255]
        img_sp[res_sp[:,3],res_sp[:,2]] = [0,255,0]

        # reference subpixel
        img_ref_sp = cv2.imread(ref_image)
        gray_ref_sp = cv2.cvtColor(img_ref_sp,cv2.COLOR_BGR2GRAY)

        # find Harris corners
        gray_ref__sp = np.float32(gray_ref_sp)
        dst_ref_sp = cv2.cornerHarris(gray_ref_sp,5,3,0.04)
        dst_ref_sp = cv2.dilate(dst_ref_sp,None)
        ret_ref_sp, dst_ref_sp = cv2.threshold(dst_ref_sp,0.01*dst_ref_sp.max(),255,0)
        dst_ref_sp = np.uint8(dst_ref_sp)

        # find centroids
        ret_ref_sp, labels_ref_sp, stats_ref_sp, centroids_ref_sp = cv2.connectedComponentsWithStats(dst_ref_sp)

        # define the criteria to stop and refine the corners
        criteria_ref = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners_ref = cv2.cornerSubPix(gray_ref_sp,np.float32(centroids_ref_sp),(5,5),(-1,-1),criteria_ref)

        # Now draw them
        res_ref_sp = np.hstack((centroids_ref_sp,corners_ref))
        res_ref_sp = np.intp(res_ref_sp)
        img_ref_sp[res_ref_sp[:,1],res_ref_sp[:,0]]=[0,0,255]
        img_ref_sp[res_ref_sp[:,3],res_ref_sp[:,2]] = [0,255,0]

        if show == '1':
            cv2.imshow('Unregistered Subpixel Calulation',img_sp)
            cv2.imshow('Reference Subpixel Calulation',img_ref_sp)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()

    # match corners
    matched_images, kp_img, kp_ref, matches = match(unreg_gray_image, ref_gray_image, unreg_read_image, ref_read_image)

    if subpixel == 1:
        matched_images_sp, kp_img_sp, kp_ref_sp, matches_sp = match(unreg_gray_image, ref_gray_image, img_sp, img_ref_sp)

    if show == '1':
        cv2.imshow('Corner Matching (Unregistered Left, Reference Right)', matched_images)
        if subpixel == 1:
            cv2.imshow(' Corner Matching(Unregistered Left, Reference Right) with subpixel enhancement', matched_images_sp)
        if cv2.waitKey(0) & 0xff == 27:
           cv2.destroyAllWindows()

    # affine transformation
    img_aligned = affine_transform(matches, unreg_read_image_2, kp_img, kp_ref)
    if subpixel == 1:
        img_aligned_sp = affine_transform(matches_sp, unreg_read_image_2, kp_img_sp, kp_ref_sp)

    if show == '1':
        cv2.imshow('Aligned Image', img_aligned)
        if subpixel == 1:
            cv2.imshow('Alignment with subpixel accuracy', img_aligned_sp)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # error calcualtion
    reg_gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    error = cv2.absdiff(reg_gray, ref_unreg)
    if subpixel == 1:
        sp_reg_gray = cv2.cvtColor(img_aligned_sp, cv2.COLOR_BGR2GRAY)
        error_sp = cv2.absdiff(sp_reg_gray, ref_unreg)
        mean_error_sp = np.mean(error_sp)
        # print(f'Mean Error with Subpixel enhancement: {mean_error_sp}')
        error_enhanced_sp = cv2.normalize(error_sp, None, 0, 255, cv2.NORM_MINMAX)
    error_enhanced = cv2.normalize(error, None, 0, 255, cv2.NORM_MINMAX)
    mean_error = np.mean(error)
    if show == '1':
        # print(f'Mean Error: {mean_error}')
        cv2.imshow('Error Image', error)
        cv2.imshow('Enhanced Error Image', error_enhanced)
        if subpixel == 1:
            cv2.imshow('Enhanced Error Image with Subpixel Accuracy', error_enhanced_sp)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # really hacky way to do this lol    
    if subpixel ==1 :
        return mean_error_sp, mean_error_sp, img_aligned, img_aligned_sp
    
    mean_error_sp = 0
    img_aligned_sp = 0
    return mean_error, mean_error_sp, img_aligned, img_aligned_sp


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    