import torch
from torchvision import transforms
from PIL import Image
import os
from model import Generator

class CycleGANTester:
    def __init__(self, model_path, default_tile_path, output_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        self.generator.eval()
        
        self.default_tile_path = default_tile_path
        self.output_dir = output_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_default_tile(self):
        image = Image.open(self.default_tile_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def generate_tiles(self, num_tiles):
        default_tile = self.load_default_tile()
        
        with torch.no_grad():
            for i in range(num_tiles):
                generated_tile = self.generator(default_tile)
                generated_tile = (generated_tile.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                output_image = Image.fromarray(generated_tile)
                output_image.save(os.path.join(self.output_dir, f'generated_tile_{i + 1}.png'))

if __name__ == "__main__":
    model_path = 'path/to/your/trained_model.pth'  # Update with your model path
    default_tile_path = 'data/default_tile.png'
    output_dir = 'output_tiles'
    
    tester = CycleGANTester(model_path, default_tile_path, output_dir)
    tester.generate_tiles(num_tiles=10)  # Generate 10 new tiles