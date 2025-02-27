import tensorflow as tf

def preprocess_image(image):
    """Preprocess an image: Resize to 256x256 and normalize to [-1, 1]."""
    image = tf.image.resize(image, [256, 256])  # Ensure image is 256x256
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

def load_dataset(data_path, batch_size=1):
    """
    Load a dataset from a directory. Assumes images are 256x256 PNGs.
    :param data_path: Path to the dataset directory.
    :param batch_size: Number of images per batch.
    :return: A tf.data.Dataset object.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode=None,  # No labels required for CycleGAN
        image_size=(256, 256),  # Resize all images to 256x256
        batch_size=batch_size
    )

    # Normalize and prefetch for performance
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
