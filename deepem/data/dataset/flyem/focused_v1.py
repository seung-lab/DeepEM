from deepem.data.dataset.flyem import cremi_b_mip1 as cremi_new
from deepem.data.dataset.flyem import focused_annotation_v1 as focused


def load_data(*args, **kwargs):
    d1 = cremi_new.load_data(*args, **kwargs)
    d2 = focused.load_data(*args, **kwargs)

    data = dict()
    data.update(d1)
    data.update(d2)
    return data
