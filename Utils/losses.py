import tensorflow as tf

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) * 0.5

def generator_loss(fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)

def cycle_consistency_loss(real_image, cycled_image, lambda_cycle=10):
    return tf.reduce_mean(tf.abs(real_image - cycled_image)) * lambda_cycle

def identity_loss(real_image, same_image, lambda_identity=0.5):
    return tf.reduce_mean(tf.abs(real_image - same_image)) * lambda_identity
