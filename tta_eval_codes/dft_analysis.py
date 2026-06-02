import os
import argparse
import numpy as np
from PIL import Image
import glob

def compute_dft_magnitude(img_gray_np):
    """
    Computes the Discrete Fourier Transform (DFT) magnitude spectrum of a grayscale image array.
    """
    # Perform 2D DFT
    dft = np.fft.fft2(img_gray_np)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum
    # +1 is used to avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    # Normalize to [0, 255] for saving/displaying
    mag_min, mag_max = np.min(magnitude_spectrum), np.max(magnitude_spectrum)
    if mag_max > mag_min:
        magnitude_spectrum = (magnitude_spectrum - mag_min) / (mag_max - mag_min) * 255.0
    else:
        magnitude_spectrum = np.zeros_like(magnitude_spectrum)
        
    return np.uint8(magnitude_spectrum)

def main():
    parser = argparse.ArgumentParser(description="Perform DFT analysis on images in a batch folder.")
    parser.add_argument("--batch_folder", type=str, required=True, help="Path to the batch folder containing images (e.g., test_result/batch1).")
    parser.add_argument("--output_folder", type=str, default=None, help="Path to save DFT results. If not specified, saves to <batch_folder>_dft.")
    parser.add_argument("--crop_pad", type=int, default=350, help="Amount of padding to crop from the right (to remove text annotations). Default is 350.")
    args = parser.parse_args()

    if not os.path.exists(args.batch_folder):
        print(f"Error: Batch folder {args.batch_folder} does not exist.")
        return

    if args.output_folder is None:
        args.output_folder = args.batch_folder.rstrip('/') + "_dft"
    
    os.makedirs(args.output_folder, exist_ok=True)

    # Gather images
    img_paths = glob.glob(os.path.join(args.batch_folder, "*.jpg")) + \
                glob.glob(os.path.join(args.batch_folder, "*.png"))
    
    if not img_paths:
        print(f"No images found in {args.batch_folder}.")
        return

    print(f"Found {len(img_paths)} images in {args.batch_folder}. Processing...")

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        try:
            img = Image.open(img_path).convert("L")  # Convert to grayscale directly
        except Exception as e:
            print(f"Warning: Could not read image {filename}. Exception: {e}. Skipping.")
            continue
            
        img_np = np.array(img)
        
        # Crop the padded text area if specified and image is wide enough
        if args.crop_pad > 0 and img_np.shape[1] > args.crop_pad:
            img_np = img_np[:, :-args.crop_pad]
            
        # Compute magnitude spectrum
        magnitude = compute_dft_magnitude(img_np)
        
        # Save output as a side-by-side image: [Original Grayscale | Magnitude Spectrum]
        combined = np.hstack((img_np, magnitude))
        
        out_path = os.path.join(args.output_folder, filename)
        Image.fromarray(combined).save(out_path)
        print(f"Saved DFT analysis for {filename} to {out_path}")

    print(f"DFT analysis complete. Results saved in {args.output_folder}")

if __name__ == "__main__":
    main()
