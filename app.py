import streamlit as st
import os
import tifffile
import imageio
import numpy as np
from PIL import Image
import cv2
from scipy.stats import kurtosis, skew, zscore
from scipy import stats

# Predefined coordinates for boxes
upper_box = [148, 58, 294, 158]
middle_box = [112, 504, 294, 620]
lower_box = [80, 982, 294, 1100]

def process_picture(image, left, upper, right, lower):
    cropped_image = image[upper:lower, left:right]
    preimage = cv2.GaussianBlur(cropped_image, (7, 7), 0)
    preimage = cv2.erode(preimage, np.ones((1, 1)), iterations=1)
    rotated_image = cv2.rotate(preimage, cv2.ROTATE_90_CLOCKWISE)
    img_norm = (rotated_image - np.min(rotated_image)) / (np.max(rotated_image) - np.min(rotated_image))
    img_stretched = np.clip((img_norm - 0.1) / 0.5, 0, 1)
    img_normalized_255 = (img_stretched * 255).astype(np.uint8)
    return img_normalized_255

def calculate_pixel_statistics(folder_path):
    files = os.listdir(folder_path)
    pixel_mean_intensity = []
    pixel_std_dev_intensity = []
    pixel_median_intensity = []
    pixel_kurtosis = []
    pixel_skewness = []
    pixel_slope = []
    pixel_z_scores = []

    for file in files:
        image_path = os.path.join(folder_path, file)
        with Image.open(image_path) as image:
            image_array = np.array(image)
            image_intensity = image_array.astype(np.uint8).flatten()

        mean_intensity = np.mean(image_intensity)
        std_dev_intensity = np.std(image_intensity)
        median_intensity = np.median(image_intensity)
        kurtosis_val = kurtosis(image_intensity)
        skewness_val = skew(image_intensity)

        x = np.arange(len(image_intensity))
        slope, _, _, _, _ = stats.linregress(x, image_intensity)

        z_scores = zscore(image_intensity)

        pixel_mean_intensity.append(mean_intensity)
        pixel_std_dev_intensity.append(std_dev_intensity)
        pixel_median_intensity.append(median_intensity)
        pixel_kurtosis.append(kurtosis_val)
        pixel_skewness.append(skewness_val)
        pixel_slope.append(slope)
        pixel_z_scores.append(z_scores)

    return (pixel_mean_intensity, pixel_std_dev_intensity, 
            pixel_median_intensity, pixel_kurtosis, 
            pixel_skewness, pixel_slope, pixel_z_scores)

def create_directories(output_folder, folder_name):
    paths = [
        os.path.join(output_folder, subfolder, folder_name, sub_subfolder)
        for subfolder in ["Cropped_Tif", "Full_Pictures", "Statistics"]
        for sub_subfolder in (["First", "Second", "Third"] if subfolder == "Cropped_Tif" else [""])
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Streamlit App

st.title('Image Processing App')

# File Input
tiff_file = st.file_uploader("Upload a TIFF file", type=["tif", "tiff"])
output_folder = st.text_input('Enter the output folder path:', '')

if tiff_file and output_folder:
    folder_path = os.path.dirname(tiff_file.name)
    files = [tiff_file.name]

    for i in files:
        if not i.endswith('.tif'):
            continue

        folder_name = i.replace('.tif', '')
        image_path = os.path.join(folder_path, i)
        create_directories(output_folder, folder_name)

        with tifffile.TiffFile(tiff_file) as tif:
            image = tif.series[0].asarray()
            height, width, num_slices = image.shape

        max_intensity = np.max(image)
        min_intensity = np.min(image)

        scaled_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 65535

        for n in range(num_slices):
            output_file_path = os.path.join(output_folder, "Full_Pictures", folder_name, f'{folder_name}_slice_{n+1}.png')
            image_slice = scaled_image[:, :, n].astype(np.uint16)
            imageio.imwrite(output_file_path, image_slice)

            image_cv = cv2.imread(output_file_path, cv2.IMREAD_UNCHANGED)
            rotated_image1 = process_picture(image_cv, *upper_box)
            rotated_image2 = process_picture(image_cv, *middle_box)
            rotated_image3 = process_picture(image_cv, *lower_box)

            cv2.imwrite(os.path.join(output_folder, "Cropped_Tif", folder_name, "First", f'{folder_name}_first_{n+1}.png'), rotated_image1)
            cv2.imwrite(os.path.join(output_folder, "Cropped_Tif", folder_name, "Second", f'{folder_name}_second_{n+1}.png'), rotated_image2)
            cv2.imwrite(os.path.join(output_folder, "Cropped_Tif", folder_name, "Third", f'{folder_name}_third_{n+1}.png'), rotated_image3)

        first_stats = calculate_pixel_statistics(os.path.join(output_folder, "Cropped_Tif", folder_name, "First"))
        second_stats = calculate_pixel_statistics(os.path.join(output_folder, "Cropped_Tif", folder_name, "Second"))
        third_stats = calculate_pixel_statistics(os.path.join(output_folder, "Cropped_Tif", folder_name, "Third"))

        first_mean_intensity, first_std_dev_intensity, first_median_intensity, first_kurtosis, first_skewness, first_slope, first_z_scores = first_stats
        second_mean_intensity, second_std_dev_intensity, second_median_intensity, second_kurtosis, second_skewness, second_slope, second_z_scores = second_stats
        third_mean_intensity, third_std_dev_intensity, third_median_intensity, third_kurtosis, third_skewness, third_slope, third_z_scores = third_stats

        # Writing statistics to text files
        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_mean_intensity.txt'), 'w') as f:
            f.write('\n'.join(map(str, first_mean_intensity)) + '\n')
            f.write('\n'.join(map(str, second_mean_intensity)) + '\n')
            f.write('\n'.join(map(str, third_mean_intensity)) + '\n')

        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_std_dev_intensity.txt'), 'w') as f:
            f.write('\n'.join(map(str, first_std_dev_intensity)) + '\n')
            f.write('\n'.join(map(str, second_std_dev_intensity)) + '\n')
            f.write('\n'.join(map(str, third_std_dev_intensity)) + '\n')

        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_median_intensity.txt'), 'w') as f:
            f.write('\n'.join(map(str, first_median_intensity)) + '\n')
            f.write('\n'.join(map(str, second_median_intensity)) + '\n')
            f.write('\n'.join(map(str, third_median_intensity)) + '\n')

        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_kurtosis.txt'), 'w') as f:
            f.write('\n'.join(map(str, first_kurtosis)) + '\n')
            f.write('\n'.join(map(str, second_kurtosis)) + '\n')
            f.write('\n'.join(map(str, third_kurtosis)) + '\n')

        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_skewness.txt'), 'w') as f:
            f.write('\n'.join(map(str, first_skewness)) + '\n')
            f.write('\n'.join(map(str, second_skewness)) + '\n')
            f.write('\n'.join(map(str, third_skewness)) + '\n')

        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_slope.txt'), 'w') as f:
            f.write('\n'.join(map(str, first_slope)) + '\n')
            f.write('\n'.join(map(str, second_slope)) + '\n')
            f.write('\n'.join(map(str, third_slope)) + '\n')

        with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_z_scores.txt'), 'w') as f:
            for scores in first_z_scores:
                f.write('\n'.join(map(str, scores)) + '\n')
            for scores in second_z_scores:
                f.write('\n'.join(map(str, scores)) + '\n')
            for scores in third_z_scores:
                f.write('\n'.join(map(str, scores)) + '\n')

    st.success('Processing complete!')
