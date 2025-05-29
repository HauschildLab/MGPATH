"""
@author : Tien Nguyen
@date   : 2024-Oct-10 
"""
import os

import argparse

import tqdm
import h5py
import timm
from PIL import Image

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class PatchesDataset(Dataset):
    def __init__(
        self,
        file_path,
        transform=None
    ) -> None:
        # file_names = os.listdir(file_path)
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
        img = img.convert("RGB")
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)

    def __del__(self):
        self.h5_file.close()


def eval_transforms(pretrained=False):
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


def extract_embeddings(
    model, 
    fname, 
    dataloader, 
    overwrite=False
) -> None:

#    if os.path.isfile('%s.h5' % fname) and (overwrite == False):
#        return None

    embeddings, coords, file_names = [], [], []

    print("=====================Extract VISION embedding========================")
    for batch, coord in tqdm.tqdm(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            embeddings.append(model(batch).detach().cpu().numpy().squeeze())
            file_names.append(coord)

    print("====================Extract Coordinates===========================")
    for file_name in tqdm.tqdm(file_names):
        for coord in file_name:
            coord = coord.rstrip('.png').split('_')
            coords.append([int(coord[0]), int(coord[1])])

    print(fname)

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    f = h5py.File(fname+'.h5', 'w')
    f['features'] = embeddings
    f['coords'] = coords
    f.close()


def create_embeddings(
    patches_dir: str,
    saved_embeddings_dir: str
) -> None:
    batch_size=64
    print("=========================Loading Model===============================")
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    tile_encoder.to(device)
    print("=========================Finished Loading Model===============================")

    preprocess_transform = eval_transforms()
    for wsi_name in tqdm.tqdm(os.listdir(patches_dir)):
        print("Processing: ", wsi_name)
        dataset = PatchesDataset(os.path.join(patches_dir, wsi_name, wsi_name + '.h5'), transform=preprocess_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=64)
        fname = os.path.join(saved_embeddings_dir, wsi_name)
        if(not os.path.exists(fname)):
            extract_embeddings(tile_encoder, fname, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for extract tile-level embedding from GigaPath')
    parser.add_argument('--saved_dir', type=str, default=None, help='saved embedding directory')
    parser.add_argument('--patches_dir', type=str, default=None, help='patches directory')
    args = parser.parse_args()

    device = torch.device('cuda')
    saved_embeddings_dir = args.saved_dir

    create_embeddings(args.patches_dir, saved_embeddings_dir)

