import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.files = self.get_image_files(root)
        self.root = root

    def get_image_files(self, root):
        files = []
        for file in os.listdir(root):
            if file.lower().endswith('.png'):
                files.append(file)
        return sorted(files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)

def get_dataloader(root, batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader