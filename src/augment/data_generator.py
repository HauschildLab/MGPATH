"""
@author : Tien Nguyen
@date   : 2024-Aug-21
"""
import os
import h5py
import numpy
from PIL import Image
from patch_data_handler import PatchesDataset

from tqdm import tqdm

class DataAugmentGenerator(object):
    def __init__(
        self,
        spatial_augmentor,
        color_augmentor
    ) -> None:
        self.spatial_augmentor = spatial_augmentor
        self.color_augmentor = color_augmentor


    def augment(
        self,
        patch
    ) -> object:
        patch = self.spatial_augmentor(numpy.array(patch))
        # patch = self.color_augmentor(Image.fromarray(patch))
        return patch


    def run(
        self,
        patches_dir,
        wsi_dir,
        saved_dir
    ) -> None:
        wsi_file = os.path.join(patches_dir, wsi_dir, wsi_dir + '.h5')
        dataset = PatchesDataset(wsi_file, transform=None)
        if not os.path.exists(os.path.join(saved_dir, wsi_dir)):
            os.makedirs(os.path.join(saved_dir, wsi_dir), exist_ok=True)
        saved_wsi_file = os.path.join(saved_dir, wsi_dir, wsi_dir + '.h5')
        with h5py.File(saved_wsi_file, 'w') as h5_file:
            augments, coords, file_names = [], [], []
            for batch, coord in tqdm(dataset):
                augmented_patch = self.augment(batch)
                patch_np = numpy.array(augmented_patch)
                patch_name = coord
                h5_file.create_dataset(patch_name, data=patch_np)
