from __future__ import print_function

from augmentor import *


def get_augmentation(is_train, grayscale=False, warping=False, misalign=False,
                     missing=0, blur=0, random=True, recompute=False, **kwargs):
    # Misalignment
    if misalign:
        # Mild misalignment
        m1 = Blend(
            [Misalign((0,10), margin=1), SlipMisalign((0,10), interp=True, margin=1)],
            props=[0.5,0.5]
        )

        # Medium misalignment
        m2 = Blend(
            [Misalign((0,20), margin=1), SlipMisalign((0,20), interp=True, margin=1)],
            props=[0.5,0.5]
        )

        # Large misalignment
        m3 = Blend(
            [Misalign((0,30), margin=1), SlipMisalign((0,30), interp=True, margin=1)],
            props=[0.5,0.5]
        )

    # Missing sections
    if missing > 0:
        m4 = Compose([
            MixedMissingSection(maxsec=1, double=True, random=random),
            MixedMissingSection(maxsec=missing, double=False, random=random)
        ])

    augs = list()

    # Recompute connected components
    if recompute:
        augs.append(Label())

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
                        density=0.3, skip=0.1, random=random)
            )

    # Grayscale
    if grayscale:
        augs.append(
            MixedGrayscale2D(
                contrast_factor=0.5,
                brightness_factor=0.5,
                prob=1, skip=0.3))

    # Missing section & misalignment
    if misalign and missing > 0:
        augs.append(Blend([
            Compose([m1,m2,m3]),
            MisalignPlusMissing((5,20), random=random),
            m4
        ]))
    elif misalign:
        augs.append(Compose([m1,m2,m3]))
    elif missing > 0:
        augs.append(m4)

    # Out-of-focus
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train and warping:
        warp = Blend([
            Warp(skip=0.3),
            Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1,
                 shear_max=0.0, stretch_max=0.0)
        ])
        augs.append(warp)

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
