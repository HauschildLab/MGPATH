# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
import numpy
import torch

from transformers import CLIPModel

from models import Adapter

class PLIPProjector(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super(PLIPProjector, self).__init__()
        print("use PLIP Projector")
        self.ImageMLP = Adapter(image = True, hidden=512)
        self.TextMLP = Adapter(image = False, hidden=512)
        self.temperature = torch.nn.Parameter(\
                    torch.tensor([numpy.log(1/0.02)]), requires_grad=True)

        self.text_model = CLIPModel.from_pretrained("vinid/plip")
