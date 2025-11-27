import tensorflow as tf
import functools
import numpy as np

from losses import gradient_penalty

def enhanced_summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
            historgram_buckets=None,
            name='summary'):
    """Summary.

    Examples
    --------
    #>>> enhanced_summary({'a': data_a, 'b': data_b})

    """
    def _summary(name, data):
        if data.shape == ():
            tf.summary.scalar(name, data, step=step)
        else:
            if 'mean' in types:
                tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
            if 'std' in types:
                tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
            if 'max' in types:
                tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
            if 'min' in types:
                tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
            if 'sparsity' in types:
                tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step)
            if 'histogram' in types:
                tf.summary.histogram(name, data, step=step, buckets=historgram_buckets)

    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)

class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)

A2B_pool = ItemPool(50)
B2A_pool = ItemPool(50)
d_loss, g_loss = losses.get_adversarial_losses_fn('lsgan') #choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan']
grad_pen = 'none' #choices=['none', 'dragan', 'wgan-gp']

@tf.function
def train_G(A, B, G_A2B, G_B2A, D_A, D_B, G_optimizer,  cycle_loss_weight = 10.0, identity_loss_weight = 0.0):
    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)
        B2A = G_B2A(B, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        ell1loss = tf.losses.MeanAbsoluteError()
        A2B_g_loss = g_loss(A2B_d_logits)
        B2A_g_loss = g_loss(B2A_d_logits)
        A2B2A_cycle_loss = ell1loss(A, A2B2A)
        B2A2B_cycle_loss = ell1loss(B, B2A2B)
        A2A_id_loss = ell1loss(A, A2A)
        B2B_id_loss = ell1loss(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_D(A, B, A2B, B2A, D_A, D_B, D_optimizer, gradient_penalty_mode=grad_pen, gradient_penalty_weight=10):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss(B_d_logits, A2B_d_logits)
        D_A_gp = gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=gradient_penalty_mode)
        D_B_gp = gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def train_step(A, B, G_A2B, G_B2A, D_A, D_B, G_optimizer, D_optimizer):
    A2B, B2A, G_loss_dict = train_G(A, B, G_A2B, G_B2A, D_A, D_B, G_optimizer)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A, D_A, D_B, D_optimizer)

    return G_loss_dict, D_loss_dict



def generate_images(model, writer, test_input, tar, step):
  prediction = model.predict(test_input)

  with writer.as_default():
      tf.summary.image("Test Input", test_input* 0.5 + 0.5,
                       max_outputs=1, step=step)
      tf.summary.image("Target", tar* 0.5 + 0.5,
                       max_outputs=1, step=step)
      tf.summary.image("Prediction", prediction* 0.5 + 0.5,
                       max_outputs=1, step=step)


def fit(train_ds, EpochsNum, writer, G_A2B, G_B2A, D_A, D_B, G_optimizer, D_optimizer, checkpoint_prefix, example_input, example_target):

    bestVal = 1e4
    tf.print("Training started ...")
    cont = 0
    for epoch in range(EpochsNum):
        for A, B in train_ds:
            G_loss_dict, D_loss_dict = train_step(A, B, G_A2B, G_B2A, D_A, D_B, G_optimizer, D_optimizer)
            cont = cont + 1

            if cont % 1000 == 0:
                tf.print(f"Epoch={epoch},Iteration numbers for Gens and Disc=({G_optimizer.iterations.numpy()},{D_optimizer.iterations.numpy()}) completed.")
                with writer.as_default():
                    # # summary
                    enhanced_summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
                    enhanced_summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')

        currentDiff = tf.reduce_mean(tf.abs(example_target - G_A2B.predict(example_input)))

        if bestVal > currentDiff:
            tf.print(f"During epoch {epoch} a new best ckp was saved.")
            bestVal = currentDiff
            G_A2B.save(checkpoint_prefix + "_Best.h5")



        generate_images(G_A2B, writer, example_input, example_target, epoch)

        if epoch % (EpochsNum//20) == 0:
            tf.print("---------------------------------------------")
            tf.print(f"Epoch {epoch} is ended. Saving a checkpoint.")
            G_A2B.save(checkpoint_prefix + "_EndOfEpoch" + str(epoch) + ".h5")

    G_A2B.save(checkpoint_prefix + "_Last.h5")
    tf.print("... training is complete.")


