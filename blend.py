import cv2
import numpy as np
import os

def blend_images(image_folder_path, output_path, image_format='jpg'):
    # Get all files' names
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(image_format)]
    
    # Imgs are enough or not
    if len(image_files) < 1:
        print("Not enough images")
        return

    # initialize
    first_image_path = os.path.join(image_folder_path, image_files[0])
    accumulated_image = cv2.imread(first_image_path).astype(np.float32)

    # sum
    for image_file in image_files[1:]:
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path).astype(np.float32)
        accumulated_image += image
    
    # avg
    blended_image = accumulated_image / len(image_files)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    # create location
    output_path = os.path.join(output_path, "blend.jpg")
    print(output_path)
    
    # save
    #cv2.imshow('blended_image', blended_image)
    cv2.imwrite(output_path, blended_image)
    print(f"Blended Reference Image Successfully!")

    return output_path