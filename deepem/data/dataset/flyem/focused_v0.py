from deepem.data.dataset.flyem import cremi_b_mip1 as cremi_new
from deepem.data.dataset.flyem import cremi_dodam_mip1 as cremi_dodam
from deepem.data.dataset.flyem import focused_annotation as focused


def load_data(*args, **kwargs):
    d1 = cremi_new.load_data(*args, **kwargs)
    d2 = cremi_dodam.load_data(*args, **kwargs)
    d3 = focused.load_data(*args, **kwargs)

    data = dict()
    data.update(d1)
    data.update(d2)
    data.update(d3)
    return data
