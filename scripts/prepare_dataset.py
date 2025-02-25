import os
import shutil
import random
import config  # Import settings from config.py

# Paths
trainA_dir = os.path.join(config.DATASET_PATH, "trainA")  # Augmented Floor_01
trainB_dir = os.path.join(config.DATASET_PATH, "trainB")  # Forest tiles (already exists)
testA_dir = os.path.join(config.DATASET_PATH, "testA")  # To be filled
testB_dir = os.path.join(config.DATASET_PATH, "testB")  # To be filled

# Ensure directories exist
os.makedirs(testA_dir, exist_ok=True)
os.makedirs(testB_dir, exist_ok=True)

# Get all forest tiles from trainB
forest_tiles = [f for f in os.listdir(trainB_dir) if f.endswith(('.png', '.jpg'))]
random.shuffle(forest_tiles)

# Ensure enough forest tiles exist
if len(forest_tiles) < config.NUM_TEST_TILES:
    raise ValueError("Not enough forest tiles in trainB for testB!")

# Pick test samples from trainB (don't copy everything again!)
for tile in forest_tiles[:config.NUM_TEST_TILES]:
    src_path = os.path.join(trainB_dir, tile)
    dst_path = os.path.join(testB_dir, tile)

    if not os.path.exists(dst_path):  # Prevent overwriting
        shutil.copy(src_path, dst_path)

# Pick test samples from trainA (augmented tiles)
augmented_tiles = [f for f in os.listdir(trainA_dir) if f.endswith('.png')]
random.shuffle(augmented_tiles)

for i, tile in enumerate(augmented_tiles[:config.NUM_TEST_TILES]):
    src_path = os.path.join(trainA_dir, tile)
    dst_path = os.path.join(testA_dir, f"testA_aug_{i}.png")

    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

print("Dataset prepared successfully!")
