# tanner, noah
# wei, jianya
# image registraiton application

import tkinter as tk
import cv2
from tkinter import filedialog
from tkinter import messagebox
from ignores.tif2jpg import tif2jpg
from main import main
from blend import blend_images
import os

# functions
def browse_reference():
    folder_selected = filedialog.askdirectory(title="Select Reference Folder")
    if folder_selected:
        reference_folder_var.set(folder_selected)

def browse_output_reference():
    folder_selected = filedialog.askdirectory(title="Select Output for Blended Image")
    if folder_selected:
        reference_output_folder_var.set(folder_selected)

def browse_input():
    folder_selected = filedialog.askdirectory(title="Select Input Image Folder")
    if folder_selected:
        input_folder_var.set(folder_selected)

def browse_output():
    folder_selected = filedialog.askdirectory(title="Select Registered Image Output Destination")
    if folder_selected:
        output_folder_var.set(folder_selected)

def browse_reference_blended():
    file_selected = filedialog.askopenfilename(title="Select blended reference image")
    if file_selected:
        reference_image_var.set(file_selected)

def submit_paths():
    reference_image = reference_image_var.get()
    input_folder = input_folder_var.get()
    output_folder = output_folder_var.get()
    if not input_folder or not output_folder or not reference_image:
        tk.messagebox.showerror("Error", "Please select both input and output folders, and ensure that reference image has been processed as well.")
    else:
        # Replace this with your logic
        process()

def blend_reference():
    # check if both reference image folder and output have been selected
    reference_location = reference_folder_var.get()
    blended_output = reference_output_folder_var.get()
    temp_dir = tif2jpg(reference_location)
    blend_image_location = blend_images(temp_dir, blended_output, 'jpg')
    messagebox.showinfo("Output Location", f"Blended image output to: {blend_image_location}")

    return blend_image_location

def process():
    # get subpixel var
    sp_enabled = sp_var.get()
    # get reference image location
    ref = reference_image_var.get()
    # get input locaiton
    input = input_folder_var.get()
    # get output location
    output = output_folder_var.get()

    # if subpixel is enabled, make subpixel folder in output
    if sp_enabled:
        try: 
            output_sp = os.path.join(output, "subpixel images")
            os.mkdir(output_sp)
        except FileExistsError:
            print(f"Subpixel output folder already exists at {output_sp}")
        try:
            output_normal = os.path.join(output, "registered images")
            os.mkdir(output_normal)
        except FileExistsError:
            print(f"Non-Subpixel folder already exists at {output_normal}")
    else:
        try:
            output_normal = os.path.join(output, "registered images")
            os.mkdir(output_normal)
        except FileExistsError:
            print(f"Non-subpixel folder folder already exists at {output_normal}")

    # call main function 
    for folder in os.listdir(input):

        # exclude 0mph folder
        folder_path = os.path.join(input, folder)
        if os.path.isdir(folder_path) and folder != "0mph":

            # Create a corresponding output folder for the current folder
            output_folder = os.path.join(output_normal, folder)  # Subfolder inside output_normal
            if sp_enabled:
                output_sp_folder = os.path.join(output_sp, folder)  # Subfolder inside output_sp (if sp_enabled)
            else: 
                output_sp_folder = None

            # Create the folders if they don't exist
            os.makedirs(output_folder, exist_ok=True)
            if sp_enabled:
                os.makedirs(output_sp_folder, exist_ok=True)

            # Initialize error files for the folder
            error_file_path = os.path.join(output_folder, "error.txt")
            sp_error_file_path = os.path.join(output_sp_folder, "sp_error.txt") if sp_enabled else None

            # Write headers for the error files (if not already created)
            with open(error_file_path, "w") as error_file:
                error_file.write(f"Errors for images in folder: {folder}\n")
            if sp_enabled:
                with open(sp_error_file_path, "w") as sp_error_file:
                    sp_error_file.write(f"Subpixel errors for images in folder: {folder}\n")
            
            # iterate through and get each image
            for root, _, files in os.walk(folder_path):
                files_to_process = [file for file in files if not file.startswith('._')]
                for file in files_to_process:
                    print(f"Processing folder: {folder}")
                    if file.lower().endswith('.tif'):
                        tif_path = os.path.join(root, file)

                        # call main processing fn
                        # show = 1, show images, show = 0, no show images
                        error, error_sp, img_aligned, img_aligned_sp = main(tif_path, ref, sp_enabled, show=0)

                        # write image errors
                        with open(error_file_path, "a") as error_file:
                            error_file.write(f"Image: {file}, Error: {error}\n")
                        
                        # write sp image errors
                        if sp_enabled:
                            with open(sp_error_file_path, "a") as sp_error_file:
                                sp_error_file.write(f"Image: {file}, Error: {error_sp}\n")
                        
                        # save the aligned image (img_aligned) to output directory
                        new_filename = f"aligned_{file[:-4]}.png"
                        aligned_path = os.path.join(output_folder, new_filename)
                        cv2.imwrite(aligned_path, img_aligned)

                        if sp_enabled:
                            new_sp_filename = f"aligned_sp_{file[:-4]}.png"
                            aligned_sp_path = os.path.join(output_sp_folder, new_sp_filename)
                            cv2.imwrite(aligned_sp_path, img_aligned_sp)

    return

# main gui
app = tk.Tk()
app.title('Image Registration App')

# reference image folder
reference_folder_var = tk.StringVar()
tk.Label(app, text="Reference Image Folder").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
tk.Entry(app, textvariable=reference_folder_var, width=30).grid(row=0, column=1, padx=10, pady=5)
tk.Button(app, text="Browse", command=browse_reference).grid(row=0, column=2, padx=10, pady=5)

# reference image output
reference_output_folder_var = tk.StringVar()
tk.Label(app, text="Processed Reference Image Output Destination:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
tk.Entry(app, textvariable=reference_output_folder_var, width=30).grid(row=1, column=1, padx=10, pady=5)
tk.Button(app, text="Browse", command=browse_output_reference).grid(row=1, column=2, padx=10, pady=5)

# process reference button
tk.Button(app, text="Process Reference Image", command=blend_reference).grid(row=2, column=1, pady=5)

# input folder
input_folder_var = tk.StringVar()
tk.Label(app, text="Unregistered Images Folder:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tk.Entry(app, textvariable=input_folder_var, width=30).grid(row=3, column=1, padx=10, pady=5)
tk.Button(app, text="Browse", command=browse_input).grid(row=3, column=2, padx=10, pady=5)

# output folder
output_folder_var = tk.StringVar()
tk.Label(app, text="Output Destination:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
tk.Entry(app, textvariable=output_folder_var, width=30).grid(row=4, column=1, padx=10, pady=5)
tk.Button(app, text="Browse", command=browse_output).grid(row=4, column=2, padx=10, pady=5)

# single, processsed reference image
reference_image_var = tk.StringVar()
tk.Label(app, text="Blended, Reference Image:").grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
tk.Entry(app, textvariable=reference_image_var, width=30).grid(row=5, column=1, padx=10, pady=5)
tk.Button(app, text="Browse", command=browse_reference_blended).grid(row=5, column=2, padx=10, pady=5)

# subpixel setting
sp_var = tk.BooleanVar()
tk.Checkbutton(app, text="Enable Subpixel Registration", variable=sp_var).grid(row=6, column=1, pady=20)

# go button
tk.Button(app, text="Register Images", command=submit_paths).grid(row=7, column=1, pady=20)

app.mainloop()