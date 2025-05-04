import numpy as np
import cv2

# load the unregistered and reference images
unreg = cv2.imread('B00018.tif', cv2.IMREAD_UNCHANGED)
reference = cv2.imread('blend.tif', cv2.IMREAD_UNCHANGED)

# Apply harris corner calc to each image
block_size = 5
aperture_size = 3
k = 0.04
# has to be np.float32() as per harrisCorner docs
dst_unreg = cv2.cornerHarris(np.float32(unreg), block_size, aperture_size, k)
dst_reference = cv2.cornerHarris(np.float32(unreg), block_size, aperture_size, k)
print
# highligth corners on the original images
unreg[dst_unreg > 0.1 * dst_unreg.max()] = [0, 0, 255]
reference[dst_reference > 0.1 * dst_reference.max()] = [0, 0, 255]

# show the harris corner calc
cv2.imshow('Harris Corner Unregistered Image', unreg)
cv2.imshow('Harris Corner Reference Image', reference)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# match the corners

# apply affine transformation to each image

# calculate error