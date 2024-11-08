import cv2
import numpy as np

# read
def main():
    image_path = 'C:/Users/JianyaWei/Desktop/0mph.jpg' 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error")
        return

    # GaussianBlur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # canny
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    cv2.imshow('Original Image', image)
    cv2.imshow('Edges Detected', edges)

    # output_path = r'C:\\Users\\JianyaWei\\Desktop\\canny_output.png'
    # print("Saving edge-detected image to:", output_path)
    # cv2.imwrite(output_path, edges)
    # print("Image saved successfully.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()