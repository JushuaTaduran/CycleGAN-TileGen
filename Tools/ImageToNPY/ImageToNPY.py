import os
import numpy as np
from PIL import Image

def convert_tileset_to_npy(tileset_dir, output_file):
  """
  Converts a directory of tile images (PNG) to a NumPy array (.npy) file.

  Args:
    tileset_dir: Path to the directory containing the tile images.
    output_file: Path to the output .npy file.
  """

  tile_images = []
  for filename in os.listdir(tileset_dir):
    if filename.endswith(".png"):
      filepath = os.path.join(tileset_dir, filename)
      img = Image.open(filepath).convert("RGBA")  # Load with alpha channel
      img_array = np.array(img) 
      tile_images.append(img_array)

  tile_array = np.array(tile_images) 
  np.save(output_file, tile_array)
  print(f"Tileset converted to {output_file}")

# Example usage:
tileset_directory = "D:\Thesis Development\Testing\GAN\Model#5\Datasets\Icy_Tileset"  # Replace with the actual path
output_filename = "icy_tileset.npy"
convert_tileset_to_npy(tileset_directory, output_filename)

loaded_tileset = np.load(output_filename)
print(f"Loaded tileset shape: {loaded_tileset.shape}") 

# Optionally, visualize a few tiles (if applicable)
import matplotlib.pyplot as plt
for i in range(min(5, loaded_tileset.shape[0])): 
  plt.imshow(loaded_tileset[i])
  plt.show()