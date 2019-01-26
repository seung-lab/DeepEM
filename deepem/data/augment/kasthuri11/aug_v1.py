from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, recompute=False, grayscale=False, blur=0,
                     warping=False, misalign=0, **kwargs):
    augs = list()

    # Misalignment
    if is_train:
        if misalign > 0:
            augs.append(
                Blend(
                    [Misalign((0,misalign)),
                     SlipMisalign((0,misalign), interp=True),
                     None],
                    props=[0.5,0.2,0.3]
                )
            )

        # Brightness & contrast purterbation
        if grayscale:
            augs.append(
                MixedGrayscale2D(
                    contrast_factor=0.5,
                    brightness_factor=0.5,
                    prob=1, skip=0.3))

        # Out-of-focus section
        if blur > 0:
            augs.append(MixedBlurrySection(maxsec=blur))

        # Warping
        if warping:
            augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0))

    # Flip & rotate
    augs.append(FlipRotate())

    # Recompute connected components
    if recompute:
        augs.append(Label())

    return Compose(augs)
