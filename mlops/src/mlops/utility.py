import os

import cv2


def imread_16U(image_filename):
    image = cv2.imread(image_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) == 2:
        image = image[..., None]
    return image


def _get_index_array_filename(image_filename, flawless_params):
    """
    """
    index_array_dir = flawless_params.pickle_dir or os.path.basename(image_filename)
    if not os.path.exists(index_array_dir):
        os.makedirs(index_array_dir)
    basename, _ = os.path.splitext(os.path.basename(image_filename))
    return os.path.join(index_array_dir, "{}_index_array.npz".format(basename))
