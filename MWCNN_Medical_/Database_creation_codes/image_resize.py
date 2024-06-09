import os
import cv2

def resize_images(input_dir, output_dir, target_size=(64, 64)):
    # Output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error reading {input_path}")
                continue

            # Resize the image
            resized_image = cv2.resize(image, target_size)

            # Save the resized image
            cv2.imwrite(output_path, resized_image)
            print(f"Saved resized image to {output_path}")

input_dir = 'blur_noisy_images'  # Change to your input directory
output_dir = 'blur_noisy_images_1200x1200'  # Change to your output directory
resize_images(input_dir, output_dir, target_size=(1200, 1200))