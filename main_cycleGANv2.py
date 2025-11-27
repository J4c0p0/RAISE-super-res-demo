

#https://github.com/LynnHo/CycleGAN-Tensorflow-2/tree/master
import tensorflow as tf

from cycleGANv2.dataLoader import make_zip_dataset
from cycleGANv2.models import ConvDiscriminator, ResnetGenerator, LinearDecay
from cycleGANv2.trainingOps_CycleGANResNet import fit
from clearml import Dataset, Task
import os
import glob

params = {
    'EPOCHS': 200,
    'BUFFER_SIZE': 500,
    'BATCH_SIZE': 1,
    'cropSize': None,
    'padTestSize0' : 128 * 3,
    'padTestSize1' : 128 * 5,
    'LR': 2e-4,
    'KERDIM': 32, #64 is the default
    'NOBLOCKS': 9, #9 is the default
    'DATASET_ID': "fed0c68d944f40c3b933788d97c92243",
}
'''
py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()
'''

task = Task.init()
task.connect(params)
task.output_uri = True

G_A2B = ResnetGenerator(input_shape=(None, None, 1),
                    output_channels=1,
                    dim=params['KERDIM'],
                    n_downsamplings=2,
                    n_blocks=params['NOBLOCKS'],
                    norm='instance_norm')
G_B2A = ResnetGenerator(input_shape=(None, None, 1),
                    output_channels=1,
                    dim=params['KERDIM'],
                    n_downsamplings=2,
                    n_blocks=params['NOBLOCKS'],
                    norm='instance_norm')
D_A = ConvDiscriminator(input_shape=(params['cropSize'], params['cropSize'], 1),
                      dim=params['KERDIM'],
                      n_downsamplings=3,
                      norm='instance_norm')
D_B = ConvDiscriminator(input_shape=(params['cropSize'], params['cropSize'], 1),
                      dim=params['KERDIM'],
                      n_downsamplings=3,
                      norm='instance_norm')

if not os.path.exists('Dataset'):
    Dataset.get(dataset_id=params['DATASET_ID']).get_mutable_local_copy('Dataset')
A_img_paths = glob.glob('./Dataset/ESAOTE_Brains/*.png')
B_img_paths = glob.glob('./Dataset/ZENODO_Brains/*.png')
train_dataset, len_dataset = make_zip_dataset(A_img_paths, B_img_paths, params['BATCH_SIZE'], 256, training=True, repeat=False)

epoch_decay=100  # epoch to start decaying learning rate
G_lr_schedule = LinearDecay(params['LR'],params['EPOCHS'] * len_dataset, len_dataset * epoch_decay)
D_lr_scheduler = LinearDecay(params['LR'],params['EPOCHS'] * len_dataset, len_dataset * epoch_decay)

G_optimizer = tf.keras.optimizers.Adam(G_lr_schedule, beta_1=0.5)
D_optimizer = tf.keras.optimizers.Adam(D_lr_scheduler, beta_1=0.5)

summary_writer = tf.summary.create_file_writer('./logs/')

def Preprocess(Img):
    Img = tf.io.decode_png(Img)
    Img = tf.expand_dims(Img[:, :, 0], axis=-1)
    Img = tf.expand_dims(Img, axis=0)
    Img = tf.image.resize_with_crop_or_pad(Img, params['padTestSize0'], params['padTestSize1'])
    return (tf.cast(Img, tf.float32) / 127.5) - 1

example_input = Preprocess(tf.io.read_file("./Dataset/Validation/sb2c41_b0045_img_00005.png"))
example_target = Preprocess(tf.io.read_file("./Dataset/Validation/Patient01280_Plane3_5_of_5.png"))

fit(train_dataset, params['EPOCHS'], summary_writer, G_A2B, G_B2A, D_A, D_B, G_optimizer, D_optimizer, "ckp", example_input, example_target)

task.flush(wait_for_uploads=True)
task.close()