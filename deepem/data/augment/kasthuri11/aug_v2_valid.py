from augmentor import *


def get_augmentation(is_train, recompute=False, grayscale=False, missing=0,
                     blur=0, warping=False, misalign=0, box=None, mip=0,
                     random=False, **kwargs):
    augs = list()

    # MIP factor
    mip_f = pow(2,mip)


    # Brightness & contrast purterbation
    if grayscale:
        augs.append(
            MixedGrayscale2D(
                contrast_factor=0.5,
                brightness_factor=0.5,
                prob=1, skip=0.3))

    # Mutually exclusive augmentations
    mutex = list()

    # Misalignment
    if misalign > 0:
        mutex.append(Blend([
                Misalign((0,misalign)),
                SlipMisalign((0,misalign), interp=True),
                None],
                props=[0.5,0.2,0.3]
        ))

    # Missing section
    if missing > 0:
        mutex.append(
            MixedMissingSection(maxsec=missing, individual=True, random=random, skip=0.1)
        )

    if misalign > 0 or missing > 0:
        augs.append(Blend(mutex))

    # Box occlusion
    if is_train:
        if box == 'fill':
            dims = (6//mip_f, 30//mip_f)
            margin = (1, 6//mip_f, 6//mip_f)
            aniso = 30/(6*mip_f)
            augs.append(
                FillBox(dims=dims, margin=margin, density=0.3, individual=True,
                        aniso=aniso, skip=0.1)
            )

    # Out-of-focus section
    if blur > 0:
        augs.append(MixedBlurrySection(maxsec=blur))

    # Warping
    if is_train:
        if warping:
            augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0))

    # Flip & rotate
    augs.append(FlipRotate())

    # Recompute connected components
    if recompute:
        augs.append(Label())

    return Compose(augs)
