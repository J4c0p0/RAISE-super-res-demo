import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_cross_entropy(generated):
  return cross_entropy(tf.ones_like(generated), generated);

def generator_loss(disc_generated_output, gen_output, target, LAMBDA):

  gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


def ell1_loss(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image - cycled_image))


def ell2_loss(real_image, cycled_image):
  return tf.reduce_mean(tf.math.pow(real_image - cycled_image,2))

def get_gan_losses_fn():
    bce = tf.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(- f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = tf.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()


def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)

    return gp