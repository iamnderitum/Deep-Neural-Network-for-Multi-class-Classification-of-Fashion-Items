import cv2
import os

def convert_grascale_to_rgb(directory_path):
    """
    Converts all grayscale images in the given directory to RGB format.
    
    Args:
    - directory_path (str): Path to the directory containing grayscale images.
    """

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith("png"):
            image = cv2.imread(os.path.join(directory_path, filename), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)

            # Check if the image is grayscale
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                # Convert to RGB by stacking the grayscale channel three times
                rgb_image = cv2.merge([image] * 3)

                # Save the RGB image with the same filename
                cv2.imwrite(os.path.join(directory_path, filename), rgb_image)

            else:
                print(f"{filename} is already in color. Skipping conversion.")
                
directory_path = "/home/iamnderitum/Desktop/Code/projects/Machine-learning/cnn/Cnn Pytorch/going_modular/data/to_pred"
convert_grascale_to_rgb(directory_path)