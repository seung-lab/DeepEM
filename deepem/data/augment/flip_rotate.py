from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, **kwargs):
    augs = list()
    augs.append(FlipRotate())
    return Compose(augs)
