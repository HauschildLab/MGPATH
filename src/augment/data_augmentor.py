"""
@author : Tien Nguyen
@date   : 2024-Aug-21
@desc   : Data augmentation is implemented based on the following paper
    - author: Faryna, K., van der Laak, J. &amp; Litjens, G..
    - year  : 2021
    - title : Tailoring automated data augmentation to H&amp;E-stained histopathology.
    - conference: Proceedings of the Fourth Conference on Medical Imaging with Deep Learning
    - available https://proceedings.mlr.press/v143/faryna21a.html.
"""

from augmenters import PassThroughAugmenter
from augmenters import FlipAugmenter
from augmenters import Rotate90Augmenter

import numpy as np
import skimage.color


class DataAugmenter(object):
    def __init__(
        self,
        augmentation_tag
    ) -> None:
        """
        @param:
            +) if `augmentation_tag == baseline`, the augmenters are included
                -) rotate90: rotate the image by 90 degree
                -) flip: flip the image
        """
        #get list of augs corresponding to the tag
        self.augmenters = define_augmenters(augmentation_tag)
        #get num of augs corresponding to the tag
        self.n_augmenters = len(self.augmenters)


    def augment(
        self,
        patch
    ) -> object:
        """
        @param:
            +) patch: numpy.ndarray, shape=(H, W, C)
        """
        #switch channel to the first dim
        patch = patch.transpose((2, 0, 1))
        #loop through a list of augs
        for k in range(self.n_augmenters):
            #select an aug and randomize its sigma
            self.augmenters[k][1].randomize()
            # t = time.time()
            patch = self.augmenters[k][1].transform(patch)
            # print('{t} took {s} logs.'.format(t=self.augmenters[k][0], s=np.log(time.time() - t)), flush=True)  # TODO

        patch = patch.transpose((1, 2, 0))

        return patch


def define_augmenters(
    augmentation_tag
) -> list[tuple[str, object]]:
    #Note that this is not the baseline from paper
    if augmentation_tag == 'baseline':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=[\
                'none', 'vertical', 'horizontal', 'both'])),
         ]

    elif augmentation_tag == 'none':

        augmenters = [
            ('none', PassThroughAugmenter())
        ]

    else:
        raise Exception('Unknown augmentation tag: {tag}'.format(\
                                                        tag=augmentation_tag))

    return augmenters

def rgb_to_gray(batch):

    new_batch = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1))
    for i in range(batch.shape[0]):
        new_batch[i, :, :, 0] = skimage.color.rgb2grey(batch[i, ...])

    new_batch = (new_batch * 255.0).astype('uint8')
    return new_batch
