import os
import cv2
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original_image, distorted_image):
    # Convert images to float32
    original_image = original_image.astype(np.float32)
    distorted_image = distorted_image.astype(np.float32)
    
    # Calculate mean squared error (MSE)
    mse = np.mean((original_image - distorted_image) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        return float('inf')
    else:
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

def calculate_ssim(original_image, distorted_image):
    # Convert images to grayscale if they are in color
    if original_image.ndim == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if distorted_image.ndim == 3:
        distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM
    ssim_value = ssim(original_image, distorted_image)
    return ssim_value

def compare_images_and_save_to_csv(folder1, folder2, output_file):
    psnr_values = []
    ssim_values = []
    
    # List image files in both folders
    images1 = os.listdir(folder1)
    images2 = os.listdir(folder2)
    
    # Find common image names
    common_images = set(images1).intersection(images2)
    
    # Iterate over common image names and calculate PSNR and SSIM
    for image_name in common_images:
        image_path1 = os.path.join(folder1, image_name)
        image_path2 = os.path.join(folder2, image_name)
        
        # Read images
        original_image = cv2.imread(image_path1)
        distorted_image = cv2.imread(image_path2)
        
        # Check if images are successfully loaded
        if original_image is None or distorted_image is None:
            print(f"Error: Unable to read one or both of the images: {image_name}")
            continue
        
        # Calculate PSNR
        psnr_value = calculate_psnr(original_image, distorted_image)
        psnr_values.append(psnr_value)
        
        # Calculate SSIM
        ssim_value = calculate_ssim(original_image, distorted_image)
        ssim_values.append(ssim_value)
    
    # Calculate average PSNR and SSIM values
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)
    
    # Save PSNR and SSIM values to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'PSNR', 'SSIM'])
        for i, (psnr_value, ssim_value) in enumerate(zip(psnr_values, ssim_values)):
            writer.writerow([f'Image_{i+1}', psnr_value, ssim_value])
        writer.writerow(['Average', average_psnr, average_ssim])
    
    print(f"PSNR and SSIM values are saved to {output_file}")
    
    return psnr_values, ssim_values, average_psnr, average_ssim

# Example usage:
folder1 = "_test_set_clean"  # İlk klasör adı
folder2 = "_test_set_noisy"  # İkinci klasör adı
output_file = 'psnr_ssim_values_before_mwcnn.csv'

psnr_values, ssim_values, average_psnr, average_ssim = compare_images_and_save_to_csv(folder1, folder2, output_file)
print("PSNR values:", psnr_values)
print("SSIM values:", ssim_values)
print("Average PSNR:", average_psnr)
print("Average SSIM:", average_ssim)
