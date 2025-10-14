"""
@author : Tien Nguyen
@date  : 2024-July-20
"""
from typing import List

import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from plip.plip import PLIP
device = torch.device('cuda')
torch.multiprocessing.set_sharing_strategy('file_system')


class PatchesDataset(Dataset):
    def __init__(
        self,
        file_path,
        transform=None
    ) -> None:
        """
        @desc:
            - file_path: path to the H5PY file
        """
        self.h5_file = h5py.File(file_path, 'r')
        file_names = list(self.h5_file.keys())
        imgs = file_names[:]
        coords = []
        for file_name in file_names:
            coords.append(file_name)
        self.imgs = imgs
        self.coords = coords
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        patch_np = self.h5_file[fn][:]
        img = Image.fromarray(patch_np)
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
            img = img['pixel_values'][0]
        return img, coord

    def __len__(self):
        return len(self.imgs)

    def __del__(self):
        """Close the HDF5 file when the dataset is deleted."""
        self.h5_file.close()


def save_embeddings(
    model,
    fname,
    #dataloader,
    #images: List[PIL.Image.Image],
    dataset,
    overwrite=False
) -> None:
    images = []
    embeddings, coords, file_names = [], [], []

    for sample in dataset:
        image, coord = sample
        images.append(image)
        file_names.append(coord)
    chunk_size = 2048
    for i in tqdm(range(0, len(images), chunk_size)):
        chunk = images[i : i + chunk_size]
        outputs = model.encode_images(chunk, batch_size=chunk_size)
        embeddings.append(outputs)

    for file_name in file_names:
        coord = file_name.rstrip('.png').split('_')
        coords.append([int(coord[0]), int(coord[1])])

    print(fname)

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    f = h5py.File(fname+'.h5', 'w')
    f['features'] = embeddings
    f['coords'] = coords
    f.close()

def create_embeddings(
    embeddings_dir,
    batch_size,
    patch_datasets='path/to/patch/datasets',
    assets_dir ='./ckpts/'
) -> None:
    model = PLIP('vinid/plip')
    for wsi_name in tqdm(os.listdir(patch_datasets)):
        wsi_h5_file = os.path.join(patch_datasets, wsi_name, wsi_name + '.h5')
        dataset = PatchesDataset(wsi_h5_file, transform=None)
        fname = os.path.join(embeddings_dir, wsi_name)
        if(not os.path.exists(fname)):
            save_embeddings(model, fname, dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(\
                        description='Configurations for feature extraction')
    parser.add_argument('--patches_path', type=str)
    parser.add_argument('--saved_path', type=str)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    patches_path = args.patches_path
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)

    create_embeddings(patch_datasets=patches_path,\
                                embeddings_dir=saved_path,
                                                    batch_size=args.batch_size)
