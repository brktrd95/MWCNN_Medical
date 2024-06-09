import os
import cv2
import pydicom
import numpy as np

def add_gaussian_noise(image, mean=12, sigma=61):
    # Generate Gaussian noise
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    # Add Gaussian noise to the image
    noisy_image = image + gauss
    
    # Clip pixel values to [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)

def add_motion_blur(image, kernel_size=60):
    # Create a horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Apply the kernel to the input image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def process_inbreast_dataset(input_directory, clean_output_directory, noisy_output_directory, mean=12, sigma=61, kernel_size=60):
    # Ensure the output directories exist
    if not os.path.exists(clean_output_directory):
        os.makedirs(clean_output_directory)
    if not os.path.exists(noisy_output_directory):
        os.makedirs(noisy_output_directory)
    
    # Process each DICOM file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.dcm'):
            input_path = os.path.join(input_directory, filename)
            
            # Read the DICOM file
            dicom_file = pydicom.dcmread(input_path)
            
            # Convert the DICOM pixel array to a numpy array
            image = dicom_file.pixel_array
            
            # Normalize the image to the range [0, 255] and convert to uint8
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert grayscale image to BGR for saving as JPEG
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Add Gaussian noise to the image
            noisy_image = add_gaussian_noise(image_bgr, mean, sigma)
            
            # Add motion blur to the noisy image
            noisy_blurred_image = add_motion_blur(noisy_image, kernel_size)
            
            # Create output paths for the clean and noisy images
            clean_output_filename = os.path.splitext(filename)[0] + '.jpg'
            noisy_output_filename = os.path.splitext(filename)[0] + '.jpg'
            clean_output_path = os.path.join(clean_output_directory, clean_output_filename)
            noisy_output_path = os.path.join(noisy_output_directory, noisy_output_filename)
            
            # Save the clean and noisy images
            cv2.imwrite(clean_output_path, image_bgr)
            cv2.imwrite(noisy_output_path, noisy_blurred_image)
            print(f"Processed {filename}")

# Define the input and output directories
input_directory = 'C:/mwcnn_try/archive/INbreast Release 1.0/AllDICOMs'
clean_output_directory = 'C:/mwcnn_try/clean_blur_noisy_images'
noisy_output_directory = 'C:/mwcnn_try/blur_noisy_images'

# Process the INbreast dataset with added Gaussian noise and motion blur
process_inbreast_dataset(input_directory, clean_output_directory, noisy_output_directory)
