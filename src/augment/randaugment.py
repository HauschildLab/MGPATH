from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
This repository is build upon RandAugment implementation
https://arxiv.org/abs/1909.13719 published here
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

"""
#Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.
AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""
import inspect
import numpy as np
import math
# from tensorflow.contrib import image as contrib_image
# from tensorflow.contrib import training as contrib_training
from PIL import Image, ImageEnhance, ImageOps
from augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from augmenters.color.hedcoloraugmenter import HedColorAugmenter
import random

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.
def hsv(image, factor):
    #print('image',image.shape)
    image=np.transpose(image,[2,0,1])
    augmentor= HsbColorAugmenter(hue_sigma_range=(-factor, factor), saturation_sigma_range=(-factor, factor), brightness_sigma_range=(0, 0))
    #To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image),[1,2,0])


def hed(image, factor):
    #print('image',image.shape)
    image=np.transpose(image,[2,0,1])
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=(-factor, factor), haematoxylin_bias_range=(-factor, factor),
                                            eosin_sigma_range=(-factor, factor), eosin_bias_range=(-factor, factor),
                                            dab_sigma_range=(-factor, factor), dab_bias_range=(-factor, factor),
                                            cutoff_range=(0.15, 0.85))
    ##To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image),[1,2,0])

def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  image = Image.fromarray(image)
  image = ImageOps.solarize(image,threshold)
  return np.asarray(image)



def color(image, factor):
  """Equivalent of PIL Color."""
  image = Image.fromarray(image)
  image = ImageEnhance.Color(image).enhance(factor)
  return np.asarray(image)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  image = Image.fromarray(image)
  image = ImageEnhance.Contrast(image).enhance(factor)
  return np.asarray(image)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  image = Image.fromarray(image)
  image = ImageEnhance.Brightness(image).enhance(factor)
  return np.asarray(image)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  image = Image.fromarray(image)
  image = ImageOps.posterize(image, bits)
  return np.asarray(image)

def rotate(image,degrees, replace):
    """Equivalent of PIL Posterize."""
    image = Image.fromarray(image)
    image =  image.rotate(angle=degrees,fillcolor =replace)
    return np.asarray(image)




def translate_x(image, pixels, replace):

    """Equivalent of PIL Translate in X dimension."""

    image = Image.fromarray(image)
    image=image.transform(image.size, Image.AFFINE, (1, 0,pixels, 0, 1, 0), fillcolor =replace)
    return np.asarray(image)

def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),fillcolor =replace)
  return np.asarray(image)

def shear_x(image, level, replace):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0), Image.BICUBIC, fillcolor =replace)
  return np.asarray(image)


def shear_y(image, level, replace):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, 0, 0,level,  1, 0), Image.BICUBIC, fillcolor =replace)
  return np.asarray(image)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D uint8 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """
  image = Image.fromarray(image)
  image =  ImageOps.autocontrast(image)
  return np.asarray(image)


def identity(image):
  """Implements Identity

  """
  return image

def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  image = Image.fromarray(image)
  image =  ImageEnhance.Sharpness(image).enhance(factor)
  return np.asarray(image)



def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  image = Image.fromarray(image)
  image =  ImageOps.equalize(image)
  return np.asarray(image)



def invert(image):
  """Inverts the image pixels."""
  return 255 - image




NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Hsv': hsv,
    'Hed': hed,
    'Identity': identity,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y

}


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  rand_cva = list([1, 0])

  should_flip = random.choice(rand_cva)

  if should_flip == 1:
      final_tensor = tensor
  else:
      final_tensor = -tensor
  return final_tensor




def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)

def _enhance_level_to_arg_hsv(level):
  return (level*0.03,)

def _enhance_level_to_arg_hed(level):
  return (level*0.03,)

def _enhance_level_to_arg_contrast(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)

def _enhance_level_to_arg_brightness(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)

def _enhance_level_to_arg_color(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)



def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def level_to_arg(hparams):
  return {
      'Identity': lambda level: (),
      'Hsv': _enhance_level_to_arg_hsv,
      'Hed': _enhance_level_to_arg_hed,
      'AutoContrast': lambda level: (),
      'Equalize': lambda level: (),
      'Invert': lambda level: (),
      'Rotate': _rotate_level_to_arg,
      'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
      'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      # pylint:disable=g-long-lambda
      'TranslateX': lambda level: _translate_level_to_arg(
          level, hparams.translate_const),
      'TranslateY': lambda level: _translate_level_to_arg(
          level, hparams.translate_const),
      # pylint:enable=g-long-lambda
  }


def _parse_policy_info(
    name,
    prob,
    level,
    replace_value,
    augmentation_hparams,
    magnitude
):
  """Return the function that corresponds to `name` and update `level` param."""
  if name=='Hsv':
      func = NAME_TO_FUNC[name]
      args = level_to_arg(augmentation_hparams)[name](magnitude)
  if name=='Hed':
      func = NAME_TO_FUNC[name]
      args = level_to_arg(augmentation_hparams)[name](magnitude)
  else:
      func = NAME_TO_FUNC[name]
      args = level_to_arg(augmentation_hparams)[name](level)

  # Check to see if prob is passed into function. This is used for operations
  # where we alter bboxes independently.
  # pytype:disable=wrong-arg-types
  params = inspect.signature(func).parameters
  # if 'prob' in inspect.getargspec(func)[0]:
  #   args = tuple([prob] + list(args))
  if 'prob' in params:
      args = tuple([prob] + list(args))
  # pytype:enable=wrong-arg-types

  # Add in replace arg if it is required for the function that is being called.
  # pytype:disable=wrong-arg-types
  params = inspect.signature(func).parameters
  if 'replace' in params:
      # assert 'replace' == inspect.getargspec(func)[0][-1]
      args = tuple(list(args) + [replace_value])
  # if 'replace' in inspect.getargspec(func)[0]:
  #   # Make sure replace is the final argument
  #   assert 'replace' == inspect.getargspec(func)[0][-1]
  #   args = tuple(list(args) + [replace_value])
  # pytype:enable=wrong-arg-types

  return (func, prob, args)


def distort_image_with_randaugment(
    image,
    num_layers,
    magnitude
):
    """Applies the RandAugment policy to `image`.
    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [1, 10].
    Returns:
    The augmented version of `image`.
    """
    #print(magnitude)
    replace_value = (128, 128, 128)#[128] * 3
    #tf.logging.info('Using RandAug.')
    # augmentation_hparams = contrib_training.HParams(cutout_const=40, translate_const=10)
    augmentation_hparams = {
        'cutout_const': 40,
        'translate_const': 10
    }
    #The 'Default' option is the H&E tailored randaugment

    # available_ops = ['Color', 'Contrast', 'Equalize', 'Identity', 'Hsv', 'Hed']
    available_ops = ['Hed']
    for layer_num in range(num_layers):
        op_to_select = np.random.randint(low=0,high=len(available_ops))
        random_magnitude = np.random.uniform(low=0, high=magnitude)
        for (i, op_name) in enumerate(available_ops):
            prob = np.random.uniform(low=0.2, high=0.8)
            func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                                replace_value, augmentation_hparams,magnitude)

            if  (i== op_to_select):

                selected_func=func
                selected_args=args
                image= selected_func(image, *selected_args)

            else:
                image=image

    return image
