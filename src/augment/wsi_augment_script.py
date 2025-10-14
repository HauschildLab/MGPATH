"""
@author : Tien Nguyen
@date   : 2024-Aug-21
"""
import os
import argparse

from tqdm import tqdm

import h5py

import numpy

from concurrent.futures import ThreadPoolExecutor

# from .augmentor import DataAugmenter
from data_augmentor import DataAugmenter
from randaugment import distort_image_with_randaugment
from patch_data_handler import PatchesDataset
from data_generator import DataAugmentGenerator
from color_augmentor import ColorAugmentor
from spatial_augmentor import SpatialAugmentor

def augment_patches(
    patches_dir,
    wsi_dir,
    saved_dir
):
    """
        @desc:
            - Augment patches in a WSI
    """
    augmentation_tag = 'baseline'
    num_layers = 3 # magnitude in randaugment
    magnitude = 1 # layers in randaugment
    spatial_augmentor = SpatialAugmentor(augmentation_tag)
    color_augmentor = ColorAugmentor()
    data_augment_generator = DataAugmentGenerator(spatial_augmentor,\
                                                            color_augmentor)
    data_augment_generator.run(patches_dir, wsi_dir, saved_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
                                        'Configurations for WSI Augmentation')
    parser.add_argument('--patches_dir', type=str, default=None,\
                                    help='data directory containing patches')
    parser.add_argument('--saved_dir', type=str, default=None,\
                                    help='directory to save augmented patches')

    args = parser.parse_args()

    patches_dir = args.patches_dir
    saved_dir = args.saved_dir

    pool = ThreadPoolExecutor(max_workers=64)
    wsi_dirs = numpy.array(os.listdir(patches_dir))

    for wsi_dir in wsi_dirs:
        pool.submit(augment_patches, patches_dir, wsi_dir, saved_dir)
    pool.shutdown(wait=True)
