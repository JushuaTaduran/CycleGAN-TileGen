import os
import tensorflow as tf
import csv


def save_generated_samples(generator, test_data, output_path, epoch, save_interval=5):
    """
    Save generated images during training for visualization.
    Saves only at specified intervals.
    """
    if (epoch + 1) % save_interval != 0:
        return  # Skip saving unless it's the specified interval

    epoch_path = os.path.join(output_path, f"epoch_{epoch + 1}")
    os.makedirs(epoch_path, exist_ok=True)
    for idx, real_image in enumerate(test_data.take(5)):  # Save first 5 samples
        generated_image = generator(real_image, training=False)
        generated_image = (generated_image[0] + 1) * 127.5  # Denormalize to [0, 255]
        generated_image = tf.cast(generated_image, tf.uint8)
        sample_path = os.path.join(epoch_path, f"sample_{idx + 1}.png")
        tf.keras.utils.save_img(sample_path, generated_image)


def train(model, dataset_x, dataset_y, epochs, output_path, checkpoint_manager):
    """Training loop for CycleGAN with optimizations for saving samples and checkpoints."""
    # Create a directory for losses.csv
    losses_file = os.path.join(output_path, "losses.csv")
    if not os.path.exists(losses_file):
        with open(losses_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "gen_g_loss", "gen_f_loss", "disc_x_loss", "disc_y_loss"])

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = {"gen_g_loss": 0.0, "gen_f_loss": 0.0, "disc_x_loss": 0.0, "disc_y_loss": 0.0}
        step_count = 0

        for real_x, real_y in tf.data.Dataset.zip((dataset_x, dataset_y)):
            losses = model.train_step((real_x, real_y))
            for key in epoch_losses:
                epoch_losses[key] += losses[key].numpy()
            step_count += 1

        # Average the losses over the epoch
        for key in epoch_losses:
            epoch_losses[key] /= step_count

        # Log and save losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: G_loss={epoch_losses['gen_g_loss']:.4f}, D_X_loss={epoch_losses['disc_x_loss']:.4f}")
            with open(losses_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, epoch_losses["gen_g_loss"], epoch_losses["gen_f_loss"], epoch_losses["disc_x_loss"], epoch_losses["disc_y_loss"]])

        # Save generated samples every 5 epochs
        save_generated_samples(model.generator_g, dataset_x, output_path, epoch, save_interval=5)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_manager.save()
            print(f"Checkpoint saved at epoch {epoch + 1}")

    print("Training Complete!")
