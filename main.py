import os
import tensorflow as tf
from Models.generator import build_generator
from Models.discriminator import build_discriminator
from Models.cyclegan import CycleGAN
from Utils.data import load_dataset
from Utils.losses import generator_loss, discriminator_loss, cycle_consistency_loss, identity_loss
from Train.train_loop import train
from datetime import datetime

# Define timestamped output paths
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = os.path.join("Generated", timestamp)
checkpoint_dir = os.path.join("Checkpoints", timestamp)

os.makedirs(output_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Define dataset paths
default_tileset_path = "Datasets/Default_Tileset"
icy_tileset_path = "Datasets/Icy_Tileset"

# Hyperparameters
epochs = 10000
batch_size = 1
lambda_cycle = 10.0
lambda_identity = 0.5

# Load datasets
dataset_x = load_dataset(default_tileset_path, batch_size=batch_size)
dataset_y = load_dataset(icy_tileset_path, batch_size=batch_size)

# Build models
generator_g = build_generator()  # Default -> Icy
generator_f = build_generator()  # Icy -> Default
discriminator_x = build_discriminator()  # Default
discriminator_y = build_discriminator()  # Icy

# Initialize CycleGAN
cyclegan = CycleGAN(generator_g, generator_f, discriminator_x, discriminator_y)

# Define optimizers
gen_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
gen_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Compile the model
cyclegan.compile(
    gen_g_optimizer=gen_g_optimizer,
    gen_f_optimizer=gen_f_optimizer,
    disc_x_optimizer=disc_x_optimizer,
    disc_y_optimizer=disc_y_optimizer,
    gen_loss=generator_loss,
    disc_loss=discriminator_loss,
    cycle_loss=lambda real, cycled: cycle_consistency_loss(real, cycled, lambda_cycle),
    identity_loss=lambda real, same: identity_loss(real, same, lambda_identity),
)

# Checkpoint management
checkpoint = tf.train.Checkpoint(generator_g=generator_g,
                                 generator_f=generator_f,
                                 discriminator_x=discriminator_x,
                                 discriminator_y=discriminator_y,
                                 gen_g_optimizer=gen_g_optimizer,
                                 gen_f_optimizer=gen_f_optimizer,
                                 disc_x_optimizer=disc_x_optimizer,
                                 disc_y_optimizer=disc_y_optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restore checkpoint if available
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"Restored from {checkpoint_manager.latest_checkpoint}")
else:
    print("Initializing from scratch.")

# Train the model
train(cyclegan, dataset_x, dataset_y, epochs, output_path, checkpoint_manager)
