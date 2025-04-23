# affine transformation function
import cv2
import numpy as np

def affine_transform(matches, unreg_read_image, kp_img, kp_ref):
    # Check input image bit depth
    print(f"Input image dtype: {unreg_read_image.dtype}")
    if unreg_read_image.dtype == np.uint8:
        print("WARNING: Input image is 8-bit. Ensure this is expected.")
    elif unreg_read_image.dtype == np.uint16:
        print("Input image is 16-bit. Proceeding with transformation.")
    else:
        print(f"Input image has unexpected dtype: {unreg_read_image.dtype}")

    # Check if enough matches are available
    if len(matches) >= 5:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in matches[:5]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches[:5]]).reshape(-1, 1, 2)

        # Calculate affine matrix
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Handle grayscale (2D) and color (3D) images
        if len(unreg_read_image.shape) == 2:  # Grayscale image
            rows, cols = unreg_read_image.shape
            img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))
        elif len(unreg_read_image.shape) == 3:  # Color image
            rows, cols, ch = unreg_read_image.shape
            img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))
        else:
            raise ValueError("Unexpected image shape: {}".format(unreg_read_image.shape))
    else:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches[:3]]).reshape(-1, 1, 2)
        print(f"LOW POINT MATCHING FOR IMAGE: {unreg_read_image}")

        # Calculate affine matrix
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Handle grayscale (2D) and color (3D) images
        if len(unreg_read_image.shape) == 2:  # Grayscale image
            rows, cols = unreg_read_image.shape
            img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))
        elif len(unreg_read_image.shape) == 3:  # Color image
            rows, cols, ch = unreg_read_image.shape
            img_aligned = cv2.warpAffine(unreg_read_image, M, (cols, rows))
        else:
            raise ValueError("Unexpected image shape: {}".format(unreg_read_image.shape))

    return img_aligned