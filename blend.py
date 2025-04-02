import cv2
import numpy as np
import os

def blend_images(image_folder_path, output_path, image_format):
    # Get all files' names
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(image_format)]
    
    # Check if there are enough images
    if len(image_files) < 1:
        print("Not enough images")
        return
    
    # Initialize with first image
    first_image_path = os.path.join(image_folder_path, image_files[0])
    # Read as 8-bit image since they're JPGs
    #accumulated_image = cv2.imread(first_image_path).astype(np.float32)
    accumulated_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    #cv2.imshow('first img', accumulated_image)
    
    print(f"First image shape: {accumulated_image.shape}, dtype: {accumulated_image.dtype}")
    print(f"First image min: {np.min(accumulated_image)}, max: {np.max(accumulated_image)}")
    print(image_files)
    
    # Sum all images
    for image_file in image_files[1:]:
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        # Check if images have same dimensions
        if image.shape != accumulated_image.shape:
            print(f"Warning: Image {image_file} has different dimensions. Resizing.")
            image = cv2.resize(image, (accumulated_image.shape[1], accumulated_image.shape[0]))
        
        accumulated_image += image
        print(f"Added {image_file}, current max value: {np.max(accumulated_image)}")
    
    # Average the images
    blended_image = accumulated_image / len(image_files)
    
    blended_image = blended_image.astype(np.uint16)  # or np.float32 if needed

    
    print(f"Final blended image shape: {blended_image.shape}, dtype: {blended_image.dtype}")
    print(f"Blended image min: {np.min(blended_image)}, max: {np.max(blended_image)}")
    
    # Create full output path with filename
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save as JPG since we're working with RGB images
    output_file = os.path.join(output_path, "blend.tif")
    
    # Save the image
    cv2.imwrite(output_file, blended_image)
    print(f"Blended Reference Image Successfully saved to {output_file}!")
    
    # Display the image
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return output_file