import os
import argparse
import h5py
import numpy as np
import pandas as pd
import pickle 
import torch

parser = argparse.ArgumentParser(description='Build graph from H5 file')
parser.add_argument('--dir_path', type=str, default=None, help='path to folder containing H5 files')
parser.add_argument('--save_dir', type=str, default=None, help='where to save pickle files')
parser.add_argument('--dataset', type=str, choices = ['Lung', 'Kidney', 'BRCA'], default='Lung', help='type of dataset')
parser.add_argument('--original_patch_size', type=int, choices = [1024, 2048], default=1024, help='original patch size before being scaled by magnification')
args = parser.parse_args()


def main(args):
    list_files = [f for f in os.listdir(args.dir_path) if os.path.isfile(os.path.join(args.dir_path, f))]

    if args.dataset == 'Lung': 
        magnification_dict = pd.read_csv('TCGA-NSCLC_mag.csv')
    elif args.dataset == 'BRCA':
        magnification_dict = pd.read_csv('TCGA-BRCA_mag.csv')
    else: 
        magnification_dict = pd.read_csv('TCGA-KIRCKIRP_mag.csv')
    magnification_dict = magnification_dict.set_index('slide_id').T.to_dict('list')

    for file in list_files: 
        filename = os.path.splitext(file)[0]
        mag = magnification_dict[filename][0]
        if mag == 20: 
            patch_size = args.original_patch_size/2
        else: 
            patch_size = args.original_patch_size

        h5_file = h5py.File(os.path.join(args.dir_path, file), 'r')
        num_patches = h5_file['features'].shape[0]

        nodes = torch.from_numpy(np.array(h5_file['features']))
        coords = tuple(map(tuple, h5_file['coords']))

        coords_dict = {}
        for i in range(num_patches): 
            coords_dict[coords[i]] = i

        edges = []
        for i in range(num_patches): 
            #left
            if coords_dict.get((coords[i][0]-patch_size, coords[i][1])): 
                j = coords_dict[(coords[i][0]-patch_size, coords[i][1])]
                edges.append([i,j])
    
            #right
            if coords_dict.get((coords[i][0]+patch_size, coords[i][1])): 
                j = coords_dict[(coords[i][0]+patch_size, coords[i][1])]
                edges.append([i,j])

            #top
            if coords_dict.get((coords[i][0], coords[i][1]-patch_size)): 
                j = coords_dict[(coords[i][0], coords[i][1]-patch_size)]
                edges.append([i,j])

            #bottom
            if coords_dict.get((coords[i][0], coords[i][1]+patch_size)):
                j = coords_dict[(coords[i][0], coords[i][1]+patch_size)]
                edges.append([i,j])

        #SAVE MATRIX FOR 'edges'
        edges = torch.tensor(edges).permute(1,0).contiguous()

        data = {'nodes': nodes, 'edges': edges}

        print('Saving graph data from slide {}'.format(filename))
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, filename + '.pickle'), 'wb') as handle: 
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main(args)
