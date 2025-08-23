import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_fft_image(image_array):
    """
    Compute the log magnitude spectrum of the FFT of an image.
    """
    f = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))
    return magnitude_spectrum

def process_images_in_directory(directory):
    os.makedirs(os.path.join(directory, "fft_outputs"), exist_ok=True)
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                gray = img.convert('L')
                image_array = np.array(gray)
                fft_magnitude = compute_fft_image(image_array)

                # Save FFT magnitude spectrum as image
                plt.figure(figsize=(5, 5))
                plt.imshow(fft_magnitude, cmap='gray')
                plt.axis('off')
                save_path = os.path.join(directory, "fft_outputs", f"{os.path.splitext(filename)[0]}_fft.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

if __name__ == "__main__":
    directory = "."  # current directory
    process_images_in_directory(directory)

