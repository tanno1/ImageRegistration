import cv2
import numpy as np

# read
def main():
    image_path = r'C:\Users\JianyaWei\Desktop\60mph.jpg'  # image path
    image = cv2.imread(image_path)

    # success or not?
    if image is None:
        print("Error")
        return

    # gray img and show img
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    gray_image_1 = np.float32(gray_image) #_1 means that it is a matrix, not a picture
    
    # Harris-Corner Detection
    block_size = 5 # Indicates the size of the neighborhood used for corner detection.
    aperture_size = 3 # Usually, it should be 3,5,7
    k = 0.04 # Usually >0.04 and <0.06
    dst = cv2.cornerHarris(gray_image_1, block_size, aperture_size, k)

    # expand point area for it's easier to see
    # dst = cv2.dilate(dst, None)

    gray_image_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    gray_image_display[dst > 0.01 * dst.max()] = [0, 0, 255] # Threshold

    cv2.imshow('Original Image', image)
    cv2.imshow('Harris Corners Detected', gray_image_display)

    # # export
    # output_path = r'C:\Users\JianyaWei\Desktop\harris_corners_output.png'
    # cv2.imwrite(output_path, image)  

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
