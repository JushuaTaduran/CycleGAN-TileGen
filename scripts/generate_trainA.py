import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2  # OpenCV for better color adjustments
import config

output_dir = os.path.join(config.DATASET_PATH, "trainA")
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Load Floor_01 image
floor_img = Image.open(config.FLOOR_TILE)

# Augmentation functions
def random_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(np.random.uniform(0.9, 1.1))  # Subtle change

def random_contrast(img):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(np.random.uniform(0.9, 1.1))  # Subtle contrast

def color_shift_to_forest(img):
    """Shifts colors toward green and brown tones safely."""
    img_cv = np.array(img, dtype=np.uint8)  # Ensure uint8 format
    img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV).astype(np.int16)  # Convert to int16 to prevent overflow

    # Adjust hue safely (shifting by -10 to +10)
    h_shift = np.random.randint(-10, 11)
    img_hsv[..., 0] = np.clip(img_hsv[..., 0] + h_shift, 0, 179)  # Hue range in OpenCV is [0, 179]

    # Slight increase in saturation for richer colors
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * np.random.uniform(0.9, 1.2), 0, 255)

    # Convert back to uint8 before converting to RGB
    img_hsv = img_hsv.astype(np.uint8)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(img_rgb)

def add_moss_texture(img):
    """ Adds organic moss-like texture using Perlin noise. """
    img_np = np.array(img, dtype=np.float32)  # Convert to float
    h, w, _ = img_np.shape

    # Generate Perlin-like noise (random pattern)
    noise = np.random.uniform(0, 40, (h, w, 1))
    noise = np.repeat(noise, 3, axis=2)  # Expand to 3 channels

    img_np = np.clip(img_np + noise, 0, 255)  # Blend noise
    return Image.fromarray(img_np.astype(np.uint8))

def add_soft_blur(img):
    """ Slightly blurs to blend textures naturally. """
    return img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))

# Generate augmented tiles
for i in range(config.NUM_AUGMENTED_TILES):
    aug_img = floor_img.copy()
    
    aug_img = random_brightness(aug_img)
    aug_img = random_contrast(aug_img)
    aug_img = color_shift_to_forest(aug_img)  # Green/brown color shift
    
    if np.random.rand() > 0.5:
        aug_img = add_moss_texture(aug_img)  # Add moss texture
    
    aug_img = add_soft_blur(aug_img)  # Smooth blending
    
    aug_img.save(os.path.join(output_dir, f"Floor_01_aug_{i}.png"))

print(f"Generated {config.NUM_AUGMENTED_TILES} augmented tiles in {output_dir}")
