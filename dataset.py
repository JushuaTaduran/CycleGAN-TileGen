import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config

class ForestDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.dir_path = os.path.join(root, mode)
        self.image_filenames = [f for f in os.listdir(self.dir_path) if f.endswith(('.png', '.PNG' , '.jpg', '.jpeg'))]

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.image_filenames[index])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image
