import cv2
import numpy as np
import sys

def harris_corner(image_mat):
    # input: gray matrix img
    # Harris-Corner Detection Parameters
    block_size = 5 # Indicates the size of the neighborhood used for corner detection.
    aperture_size = 3 # Usually, it should be 3,5,7
    k = 0.04 # Usually >0.04 and <0.06
    dst = cv2.cornerHarris(image_mat, block_size, aperture_size, k)

    # expand the size of the corner indicators
    # dst = cv2.dilate(dst, None)

    # return harris corner image
    return dst

# # cli usage
# if __name__ == '__main__':
#     image = sys.argv[1]
#     read_image = cv2.imread(image)
#     gray_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
#     gray_mat = np.float32(gray_image)
#     dst = harris_corner(gray_mat)
#     # threshold is middle value 
#     read_image[dst > 0.05 * dst.max()] = [0, 0, 255]
#     print(type(read_image))
#     cv2.imshow('dst', read_image)
#     cv2.imshow('dst2', dst)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
