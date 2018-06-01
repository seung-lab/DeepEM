from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, box=None, random=True, **kwargs):
    # Mild misalignment
    m1 = Blend(
        [Misalign((0,10), margin=1), SlipMisalign((0,10), margin=1)],
        props=[0.7,0.3]
    )

    # Medium misalignment
    m2 = Blend(
        [Misalign((0,30), margin=1), SlipMisalign((0,30), margin=1)],
        props=[0.7,0.3]
    )

    # Large misalignment
    m3 = Blend(
        [Misalign((0,50), margin=1), SlipMisalign((0,50), margin=1)],
        props=[0.7,0.3]
    )

    # Missing section
    missing = Compose([
        MixedMissingSection(maxsec=1, double=True, random=random),
        MixedMissingSection(maxsec=7, double=False, random=random)
    ])

    # Lost section
    lost = Blend([
        Compose([LostSection(1), LostSection(2)]),
        LostPlusMissing(random=random)
    ])

    augs = list()

    # Box
    if is_train:
        if box == 'noise':
            augs.append(
                NoiseBox(sigma=(1,3), dims=(10,50), margin=(1,10,10),
                         density=0.3, skip=0.1)
            )
        elif box == 'fill':
            augs.append(
                FillBox(dims=(10,50), margin=(1,10,10),
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
        MisalignPlusMissing((5,30), random=random),
        missing,
        lost
    ]))

    # Out-of-focus
    augs.append(MixedBlurrySection(maxsec=7))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3))

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
