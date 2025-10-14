"""
@author : Tien Nguyen
@date   : 2024-Aug-22
"""

from augmenters import PassThroughAugmenter
from augmenters import FlipAugmenter
from augmenters import Rotate90Augmenter

class SpatialAugmentor(object):
    def __init__(
        self,
        augmentation_tag
    ) -> None:
        self.augmentation_tag = augmentation_tag
        self.augmenters = define_augmenters(augmentation_tag)
        self.n_augmenters = len(self.augmenters)

    def __call__(
        self,
        patch
    ) -> object:
        return self.augment(patch)

    def augment(self, patch):
        patch = patch.transpose((2, 0, 1))
        for k in range(self.n_augmenters):
            self.augmenters[k][1].randomize()
            patch = self.augmenters[k][1].transform(patch)

        patch = patch.transpose((1, 2, 0))
        return patch

def define_augmenters(
    augmentation_tag
) -> list[tuple[str, object]]:
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
