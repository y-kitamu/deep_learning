import sys

import cv2
import numpy as np
import tensorflow as tf
from numba import cuda


def clear_gpu(gpu_id=0):
    cuda.select_device(gpu_id)
    device = cuda.get_current_device()
    device.reset()
    cuda.close()
    print("CUDA memory released: GPU {}".format(gpu_id))


def set_gpu(gpu_id=0):
    if gpu_id < 0:
        tf.config.set_visible_devices([], 'GPU')
        return
    clear_gpu(gpu_id)
    if tf.__version__ >= "2.1.0":
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    elif tf.__version__ >= "2.0.0":
        #TF2.0
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(
            visible_device_list=str(gpu_id),  # specify GPU number
            allow_growth=True))
        set_session(tf.Session(config=config))


def tensor_to_nparray(tensor):
    """tf.Tensorを8bitの画像のnp.arrayに変換
    Args:
        Tensor (tf.Tensor) : 2, 3 or 4D image tensor.
    Return:
        arr (np.ndarray) :
    """
    arr = np.array(tensor * 255.0).astype(np.uint8)
    if np.ndim(arr) > 3:
        arr = arr[0]
    return arr[:, :, ::-1]


def read_image_to_array(image_path):
    """画像を読み込み、0 ~ 1の値のnp.array (float32)に変換
    Args:
        image_path (str)
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load content image from {}".format(image_path))
        print("Stop processing.")
        sys.exit(1)
    image = (image[None, :, :, ::-1] / 255.0).astype(np.float32)
    return image
