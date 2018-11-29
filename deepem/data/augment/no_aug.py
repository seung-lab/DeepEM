from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, recompute=False, **kwargs):
    augs = list()

    # Recompute connected components
    if recompute:
        augs.append(Label())

    # Flip & rotate
    if not is_train:
        augs.append(FlipRotate())

    return Compose(augs)
