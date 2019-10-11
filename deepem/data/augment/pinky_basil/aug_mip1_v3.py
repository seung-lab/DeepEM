from augmentor import *


def get_augmentation(is_train, box=None, missing=7, blur=7, lost=True,
                     random=False, **kwargs):
    augs = list()

    # Box
    if is_train:
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

    # Brightness & contrast purterbation
    augs.append(
        MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Missing section & misalignment
    to_blend = list()
    # Misalingments
    trans = Compose([Misalign((0, 5), margin=1),
                     Misalign((0,15), margin=1),
                     Misalign((0,25), margin=1)])

    # Out-of-alignments
    slip = Compose([SlipMisalign((0, 5), interp=True, margin=1),
                    SlipMisalign((0,15), interp=True, margin=1),
                    SlipMisalign((0,25), interp=True, margin=1)])
    to_blend.append(Blend([trans,slip], props=[0.7,0.3]))
    if is_train:
        to_blend.append(Blend([
            MisalignPlusMissing((3,15), value=0, random=random),
            MisalignPlusMissing((3,15), value=0, random=False)
        ]))
    else:
        to_blend.append(MisalignPlusMissing((3,15), value=0, random=False))
    if missing > 0:
        if is_train:
            to_blend.append(Blend([
                MixedMissingSection(maxsec=missing, individual=True, value=0, random=False),
                MixedMissingSection(maxsec=missing, individual=True, value=0, random=random),
                MissingSection(maxsec=missing, individual=False, value=0, random=random),
            ]))
        else:
            to_blend.append(
                MixedMissingSection(maxsec=missing, individual=True, value=0, random=False)
            )
    if lost:
        if is_train:
            to_blend.append(Blend([
                LostSection(1),
                LostPlusMissing(value=0, random=random),
                LostPlusMissing(value=0, random=False)
            ]))
    augs.append(Blend(to_blend))

    # Out-of-focus
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0, scale_max=1.1))

    # Flip & rotate
    augs.append(FlipRotate())

    return Compose(augs)
