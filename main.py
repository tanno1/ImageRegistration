# image registration main file
# noah tanner, jianya wei
# fall 2024

import cv2
import test_2_full
import numpy as np
from harris2 import harris_corner
from match import match
from affinetransform import affine_transform
import sys

def main(unreg_image, ref_image, sp, show, k, block, aperture):
    # enable or disable subpixel, 1 = yes, 0 = no subpixel
    # print(show)
    subpixel = sp

    # Open and read each image
    unreg = cv2.imread(unreg_image, cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(ref_image, cv2.IMREAD_UNCHANGED)

    # Ensure images are single-channel grayscale
    if len(unreg.shape) == 2:  # Already grayscale
        gray_unreg = unreg.astype(np.uint8)  # Ensure correct dtype
    else:  # Convert to grayscale
        gray_unreg = cv2.cvtColor(unreg, cv2.COLOR_BGR2GRAY)

    if len(ref.shape) == 2:  # Already grayscale
        gray_ref = ref.astype(np.uint8)  # Ensure correct dtype
    else:  # Convert to grayscale
        gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Use the grayscale images directly for further processing
    unreg_gray_image = np.float32(gray_unreg)
    ref_gray_image = np.float32(gray_ref)

    # Apply Harris corner calculations
    unreg_dst = harris_corner(unreg_gray_image, k, block, aperture)
    ref_dst = harris_corner(ref_gray_image, k, block, aperture)

    # Highlight corners on the original images
    if len(unreg.shape) == 3:  # If the original image is color
        unreg[unreg_dst > .7 * unreg_dst.max()] = [0, 0, 255]
    if len(ref.shape) == 3:  # If the original image is color
        ref[ref_dst > 0.7 * ref_dst.max()] = [0, 0, 255]

    if show == '1':
        cv2.imshow('Harris Corner Unregistered Image', unreg)
        cv2.imshow('Harris Corner Reference Image', ref)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # subpixel accuracy
    ## comment out until line 69 if no subpixel
    if subpixel == 1:
        img_sp = cv2.imread(unreg_image)
        gray_sp = cv2.cvtColor(img_sp,cv2.COLOR_BGR2GRAY)

        # find Harris corners
        gray_sp = np.float32(gray_sp)
        dst_sp = cv2.cornerHarris(gray_sp,block,aperture,k)
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
        dst_ref_sp = cv2.cornerHarris(gray_ref_sp,block,aperture,k)
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
    matched_images, kp_img, kp_ref, matches = match(unreg_gray_image, ref_gray_image, unreg, ref)

    if subpixel == 1:
        matched_images_sp, kp_img_sp, kp_ref_sp, matches_sp = match(unreg_gray_image, ref_gray_image, img_sp, img_ref_sp)

    if show == '1':
        cv2.imshow('Corner Matching (Unregistered Left, Reference Right)', matched_images)
        if subpixel == 1:
            cv2.imshow(' Corner Matching(Unregistered Left, Reference Right) with subpixel enhancement', matched_images_sp)
        if cv2.waitKey(0) & 0xff == 27:
           cv2.destroyAllWindows()

    # affine transformation
    img_aligned = affine_transform(matches, unreg, kp_img, kp_ref)
    if subpixel == 1:
        img_aligned_sp = affine_transform(matches_sp, unreg, kp_img_sp, kp_ref_sp)

    if show == '1':
        cv2.imshow('Aligned Image', img_aligned)
        if subpixel == 1:
            cv2.imshow('Alignment with subpixel accuracy', img_aligned_sp)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # Error calculation
    if len(img_aligned.shape) == 2:  # Already grayscale
        reg_gray = img_aligned
    else:  # Convert to grayscale
        reg_gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)

    # Normalize images to the range [0, 1] before calculating the absolute difference
    reg_gray = reg_gray / reg_gray.max() if reg_gray.max() > 1 else reg_gray
    gray_ref = gray_ref / gray_ref.max() if gray_ref.max() > 1 else gray_ref

    # Ensure both images have the same dtype
    reg_gray = reg_gray.astype(np.float32)
    gray_ref = gray_ref.astype(np.float32)

    # Calculate the absolute difference
    error = cv2.absdiff(reg_gray, gray_ref)

    if subpixel == 1:
        # Ensure sp_reg_gray is initialized
        if len(img_aligned_sp.shape) == 2:  # Already grayscale
            sp_reg_gray = img_aligned_sp
        else:  # Convert to grayscale
            sp_reg_gray = cv2.cvtColor(img_aligned_sp, cv2.COLOR_BGR2GRAY)

        # Normalize sp_reg_gray
        sp_reg_gray = sp_reg_gray / sp_reg_gray.max() if sp_reg_gray.max() > 1 else sp_reg_gray
        sp_reg_gray = sp_reg_gray.astype(np.float32)

        # Calculate the absolute difference for subpixel
        error_sp = cv2.absdiff(sp_reg_gray, gray_ref)
        mean_error_sp = np.mean(error_sp)
        error_enhanced_sp = cv2.normalize(error_sp, None, 0, 255, cv2.NORM_MINMAX)

    # Normalize and calculate mean error
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
