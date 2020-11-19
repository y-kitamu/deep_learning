import numpy as np
import cv2

from mlops.utility import imread_16U, _get_index_array_filename


def create_index_array_of_background_and_edge(image_filename, flawless_params, roi_size=(48, 48)):
    """Create temporaly .pkl object that is used while training.
    This function create <basename of image_filename>_index_array.npz which has two arrays named
    'edge' and 'background'. Each array represent the location(x, y) of edge (or background)
    in the image and shape is [num of edge or background, 2(x, y)].

    Args:
        image_filename (str) : Path to target image of which edge mask and background mask is created.
        flawless_params (FlawlessParameters) : hydra configuration dictionary
            that must have keys of 'max_thresh' and 'variance_thresh'.
        output_dir (str) : Path to directory in which .pkl is saved. If None, .pkl is saved in
            the same directory as 'image_filename'
        roi_size (tupe of int) : 2D tuple of (roi_width, roi_height).
    Return:
        str : Created .pkl filename
    """
    image = imread_16U(image_filename)

    masked_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                                    np.ones((int(roi_size[0] / 2), int(roi_size[1] / 2)), np.uint8))
    candidate_indices = np.where(masked_image.sum(axis=-1) != 0)
    num_candidate = len(candidate_indices[0])

    # TODO: faster (fix naive implement)
    background_list = []
    edge_list = []
    offset_x, offset_y = int(roi_size[0] / 2), int(roi_size[1] / 2)
    for idx in range(num_candidate):
        y = candidate_indices[idx][0]
        x = candidate_indices[idx][1]
        roi = image[y - offset_y:y + offset_y, x - offset_x:x + offset_x]
        if _judge_edge(roi, flawless_params):
            edge_list.append([x, y])
        else:
            background_list.append([x, y])

    background_array = np.array(background_list, dtype=int)
    edge_array = np.array(edge_list, dtype=int)
    output_fname = _get_index_array_filename(image_filename, flawless_params)
    np.savez(output_fname, background=background_array, edge=edge_array)

    return output_fname


def _judge_edge(roi, flawless_params):
    """Judge whether roi is edge or not.
    Args:
        roi (np.ndarray) :
        flawless_params (FlawlessParameters) : hydra configuration dictionary
            that must have key of 'max_thresh' and 'variance_thresh'
    Return:
        bool : If true, roi is edge.
    """
    return roi.max() > flawless_params.max_thresh and roi.var() > flawless_params.variance_thresh


def create_filelist_csv(data_root_dir, output_fname):
    """Create filename, class index list of training image.
    Args:

    """
    pass


def load_filelist_csv(csv_filename):
    """Load csv file created by `create_filelist_csv`
    Args:
        csv_filename (str) :

    """
    pass
