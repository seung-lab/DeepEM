from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, box=None, **kwargs):
    # Mild misalignment
    m1 = Blend(
        [Misalign((0,5), margin=1), SlipMisalign((0,5), margin=1)],
        props=[0.7,0.3]
    )
    # Medium misalignment
    m2 = Blend(
        [Misalign((0,15), margin=1), SlipMisalign((0,15), margin=1)],
        props=[0.7,0.3]
    )
    # Large misalignment
    m3 = Blend(
        [Misalign((0,25), margin=1), SlipMisalign((0,25), margin=1)],
        props=[0.7,0.3]
    )
    # Missing section
    missing = Compose([
        MixedMissingSection(maxsec=1, double=True, random=True),
        MixedMissingSection(maxsec=7, double=False, random=True)
    ])

    augs = list()

    # Box
    if box == 'noise':
        augs.append(
            NoiseBox(sigma=(1,3), dims=(5,25), margin=(1,5,5),
                     density=0.3, skip=0.1)
        )
    elif box == 'fill':
        augs.append(
            FillBox(dims=(5,25), margin=(1,5,5),
                    density=0.3, skip=0.1)
        )

    # Grayscale
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Missing section & misalignment
    augs.append(Blend([
        Compose([m1,m2,m3]),
        MisalignPlusMissing((3,15), random=True),
        missing
    ]))

    # Out-of-focus
    augs.append(MixedBlurrySection(maxsec=7))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3))

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
