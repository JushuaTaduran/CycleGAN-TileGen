import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the data
data_path = r'C:\Users\tadur\Downloads\environment\environment\environment_image.npy' 
images = np.load(data_path)

# Check data shape and type
print("Data shape:", images.shape)
print("Data type:", images.dtype)

# Create a directory to save the PNG images
import os
os.makedirs('png_images', exist_ok=True)

# Iterate through the images and save them as PNG files
for i, image in enumerate(images):
    # Assuming the 4th channel is irrelevant (if not, adjust accordingly)
    image_rgb = image[:, :, :, :3] 

    # Convert to uint8 for PIL
    image_uint8 = (image_rgb * 255).astype(np.uint8) 

    # Create a PIL Image object
    pil_image = Image.fromarray(image_uint8) 

    # Save the image
    filename = f'png_images/image_{i}.png'
    pil_image.save(filename)

print("Images saved as PNG files in the 'png_images' directory.")

# Optional: Display a sample image using matplotlib
plt.imshow(image_rgb) 
plt.title(f"Sample Image {i}")
plt.show()