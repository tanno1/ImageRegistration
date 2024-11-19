import os
from PIL import Image
import tempfile

# path

def tif2jpg(input_folder):
    # create temp directory to store the converted images
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_path = os.path.join(root, file)
                # load tif
                try: 
                    with Image.open(tif_path) as img:
                        # transfer RGB
                        rgb_img = img.convert("RGB")
                        # JPG path
                        filename = os.path.splitext(file)[0] + ".jpg"
                        jpg_path = os.path.join(temp_dir, filename)
                        # save as JPG
                        rgb_img.save(jpg_path, "JPEG")
                        print(f" {filename} transfered to {filename}")
                except Exception as e:
                    print(f"Error converting {tif_path}: e")
    print("Transfer Complete!")
    # return location of output files
    return temp_dir