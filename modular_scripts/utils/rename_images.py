import os

def rename_images(directory, product_type):
    #List all files in the directory
    files = os.listdir(directory)

    image_files = [file for file in files if file.endswith(".jpg") or file.endswith(".png") or file.endswith("jpeg")]

    # Initialize a counter for the numbering
    count = 20

    # Iterate over each image file
    for filename in image_files:
        new_filename = f"{product_type}-{count}"

        # Get the file extension
        _, extension = os.path.splitext(filename)

        # Construct the new full path
        new_path = os.path.join(directory, new_filename + extension)

        # Rename the file
        print(f"Renaming:{filename} to {new_filename} ")
        os.rename(os.path.join(directory, filename), new_path)
        count += 1

    print(f"Finished renaming:{count} of {product_type} ")

product_type = "shoes"
directory_path = "/home/iamnderitum/Downloads/fashion images"
rename_images(directory_path, product_type)