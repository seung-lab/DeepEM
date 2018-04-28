from __future__ import print_function

from augmentor import *


def get_augmentation(phase, **kwargs):
    augs = list()
    augs.append(FlipRotate())
    return Compose(augs)
