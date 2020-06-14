from deepem.data.dataset.flyem import cremi_b_mip1 as cremi_new
from deepem.data.dataset.flyem import focused_annotation as focused
from deepem.data.dataset.flyem import sparse_annotation as sparse


def load_data(*args, **kwargs):
    d1 = cremi_new.load_data(*args, **kwargs)
    d2 = focused.load_data(*args, **kwargs)
    d3 = sparse.load_data(*args, **kwargs)

    data = dict()
    data.update(d1)
    data.update(d2)
    data.update(d3)
    return data
