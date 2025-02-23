<<<<<<< HEAD
# CycleGAN for Tile Generation

This project implements a CycleGAN to generate 32x32 pixel tiles for a Unity game. The model is trained using a single default tile image and a dataset of 300 forest tile images.

## Project Structure

```
cycleGAN-tiles
├── data
│   ├── default_tile.png        # Single default tile image
│   ├── forest_tiles            # Directory containing forest tile images
│   │   ├── tile1.png
│   │   ├── tile2.png
│   │   └── ... (298 more tiles)
├── src
│   ├── dataset.py              # Handles loading and preprocessing of images
│   ├── model.py                # Defines the generator and discriminator models
│   ├── train.py                # Contains the training loop for the CycleGAN
│   ├── test.py                 # Evaluates the trained model and generates new tiles
│   └── utils.py                # Utility functions for image processing
├── requirements.txt            # Lists project dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cycleGAN-tiles
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your default tile image in the `data` directory as `default_tile.png`.
   - Place your forest tile images in the `data/forest_tiles` directory.

## Usage

1. **Training the Model**:
   - Run the training script:
     ```
     python src/train.py
     ```

2. **Testing the Model**:
   - After training, generate new tiles using:
     ```
     python src/test.py
     ```

## Overview of CycleGAN Implementation

The CycleGAN architecture consists of two main components: the generator and the discriminator. The generator learns to transform images from the source domain (default tile) to the target domain (forest tiles) and vice versa. The discriminator evaluates the authenticity of the generated images.

This implementation includes:
- A dataset class for loading and preprocessing images.
- Model definitions for the generator and discriminator.
- A training loop that optimizes the model parameters.
- Evaluation functions to generate and save new tile images.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
=======
# CycleGAN-TileGen
>>>>>>> feb12a975bacc31baccb4eea9a0364ed83db9150
