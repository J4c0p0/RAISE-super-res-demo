import tensorflow as tf
import multiprocessing

def normalize(image):
  return (image / 127.5) - 1

def random_gamma(image):
    gamma = tf.random.uniform((1,), 0.3, 1.7)
    image = tf.math.sign(image) * tf.math.pow(tf.math.abs(image), gamma)
    return image

def noise_injection(image):
    noise_std = tf.random.uniform([1], minval=0.0001, maxval=0.06)[0]
    noise_map = tf.random.normal(tf.shape(image), mean=0, stddev=noise_std)
    image = tf.math.sign(image) * (tf.math.abs(image) + noise_map)
    return image

def random_flip(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
    return image


@tf.function()
def augmentation(image):
    image = normalize(image)
    image = random_flip(image)
    image = random_gamma(image)
    image = noise_injection(image)
    return image

def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset

def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


def disk_image_batch_dataset(img_paths,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    """
    if labels is None:
        memory_data = img_paths
    else:
        memory_data = (img_paths, labels)

    def parse_fn(path, *label):
        Img = tf.io.read_file(path)
        Img = tf.io.decode_png(Img)
        Img = tf.expand_dims(Img[:, :, 0], axis=-1)
        Img = tf.cast(Img, tf.float32)
        return (Img,) + label

    if map_fn:  # fuse `map_fn` and `parse_fn`
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:
        map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)

    return dataset

def make_dataset(img_paths, batch_size, params, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            if params['cropSize'] != None:
                img = tf.image.resize_with_crop_or_pad(img, params['cropSize'], params['cropSize'])
            img = augmentation(img)
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize_with_crop_or_pad(img, params['padTestSize0'], params['padTestSize1'])
            img = normalize(img)
            return img

    return disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=augmentation,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset