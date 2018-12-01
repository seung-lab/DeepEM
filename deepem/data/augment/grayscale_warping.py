from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, recompute=False, grayscale=False, warping=False,
                     **kwargs):
    augs = list()

    # Recompute connected components
    if recompute:
        augs.append(Label())

    # Brightness & contrast purterbation
    if is_train and grayscale:
        augs.append(
            MixedGrayscale2D(
                contrast_factor=0.5,
                brightness_factor=0.5,
                prob=1, skip=0.3))

    # Warping
    if is_train and warping:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0))

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
