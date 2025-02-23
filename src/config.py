import torch

class Config:
    def __init__(self):
        self.image_size = 32
        self.batch_size = 1
        self.num_epochs = 600
        self.learning_rate = 0.00005
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5 * self.lambda_cycle
        self.dataset_path = 'datasets'
        self.default_tile_path = 'datasets/default_tiles/'
        self.forest_tiles_path = 'datasets/forest_tiles/Floor_01'
        self.output_path = 'output/samples/Floor_01'
        self.checkpoint_path = 'checkpoints'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_sample_interval = 5  # Save samples every 5 epochs

config = Config()