"""
@author : Tien Nguyen
@date   : 2024-Aug-22
"""

from randaugment import distort_image_with_randaugment

class ColorAugmentor(object):
    def __init__(
        self
    ) -> None:
        self.num_layers = 3
        self.magnitude = 1

    def __call__(
        self,
        patch
    ) -> object:
        return distort_image_with_randaugment(patch,\
                                        num_layers=self.num_layers,\
                                                    magnitude=self.magnitude)
