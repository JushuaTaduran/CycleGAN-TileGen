import tensorflow as tf

class CycleGAN(tf.keras.Model):
    def __init__(self, generator_g, generator_f, discriminator_x, discriminator_y):
        super(CycleGAN, self).__init__()
        self.generator_g = generator_g  # G: Default -> Icy
        self.generator_f = generator_f  # F: Icy -> Default
        self.discriminator_x = discriminator_x  # D_X: Default
        self.discriminator_y = discriminator_y  # D_Y: Icy

    def compile(self, gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer, gen_loss, disc_loss, cycle_loss, identity_loss):
        super(CycleGAN, self).compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.cycle_loss = cycle_loss
        self.identity_loss = identity_loss

    def train_step(self, data):
        # Unpack data
        real_x, real_y = data

        with tf.GradientTape(persistent=True) as tape:
            # Forward cycle: X -> Y -> X
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            # Backward cycle: Y -> X -> Y
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # Identity mapping
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            # Discriminator output
            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # Calculate losses
            gen_g_loss = self.gen_loss(disc_fake_y)
            gen_f_loss = self.gen_loss(disc_fake_x)

            cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)
            id_loss = self.identity_loss(real_x, same_x) + self.identity_loss(real_y, same_y)

            total_gen_g_loss = gen_g_loss + cycle_loss + id_loss
            total_gen_f_loss = gen_f_loss + cycle_loss + id_loss

            disc_x_loss = self.disc_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.disc_loss(disc_real_y, disc_fake_y)

        # Apply gradients
        gradients_gen_g = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        gradients_gen_f = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        gradients_disc_x = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        gradients_disc_y = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        self.gen_g_optimizer.apply_gradients(zip(gradients_gen_g, self.generator_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(gradients_gen_f, self.generator_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(gradients_disc_x, self.discriminator_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(gradients_disc_y, self.discriminator_y.trainable_variables))

        return {
            "gen_g_loss": total_gen_g_loss,
            "gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss,
        }
