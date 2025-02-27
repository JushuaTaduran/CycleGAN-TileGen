import os
from PIL import Image

def cut_image(image_path, output_dir, tile_size=(32, 32)):
    """
    Cuts an image into smaller tiles of specified size.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save the output tiles.
        tile_size: Tuple of (width, height) for each tile.
    """
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        tile_width, tile_height = tile_size

        # Create tiles
        for top in range(0, img_height, tile_height):
            for left in range(0, img_width, tile_width):
                box = (left, top, left + tile_width, top + tile_height)
                tile = img.crop(box)

                tile_name = f"tile_{top}_{left}.png"
                tile.save(os.path.join(output_dir, tile_name))

        print(f"Successfully cut image: {os.path.basename(image_path)}")
    except PermissionError:
        print(f"Error: Permission denied for accessing '{image_path}'.")
        print("Please check file permissions or run the script with administrator privileges.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Correct usage example
input_folder = r"D:\Thesis Development\Testing\GAN\Model#6\Datasets"
output_folder = r"D:\Thesis Development\Testing\GAN\Model#6\Datasets\Default_Tilesets"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all images in the input folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(input_folder, file_name)
        cut_image(image_path, output_folder)
