"""
"""
import h5py
from torch.utils.data.dataset import Dataset
from PIL import Image


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
        #img = Image.open(fn).convert('RGB')
        img = Image.fromarray(patch_np)
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)

    def __del__(self):
        """Close the HDF5 file when the dataset is deleted."""
        self.h5_file.close()
