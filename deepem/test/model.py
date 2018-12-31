from __future__ import print_function

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from deepem.utils import torch_utils
from deepem.test.mask import PatchMask, AffinityMask


class Model(nn.Module):
    """
    Model wrapper for inference.
    """
    def __init__(self, model, opt):
        super(Model, self).__init__()
        self.model = model
        self.in_spec = dict(opt.in_spec)
        self.pretrain = opt.pretrain
        self.cropsz = opt.cropsz

        # Precomputed mask
        self.mask = dict()
        if opt.blend == 'precomputed':
            for k, v in opt.scan_spec.items():
                patch_sz = v[-3:]
                if k == 'affinity':
                    edges = [(0,0,1),(0,1,0),(1,0,0)]
                    mask = AffinityMask(patch_sz, opt.overlap, edges, opt.bump)
                else:
                    mask = PatchMask(patch_sz, opt.overlap)
                    mask = np.expand_dims(mask, axis=0)
                mask = np.expand_dims(mask, axis=0)
                self.mask[k] = torch.from_numpy(mask).cuda()

    def forward(self, sample):
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = self.model(*inputs)
        outputs = dict()
        for k, x in preds.items():
            outputs[k] = F.sigmoid(x)

            # Precomputed mask
            if k in self.mask:
                outputs[k] *= self.mask[k]

            # Crop outputs.
            if self.cropsz is not None:
                outputs[k] = torch_utils.crop_border(outputs[k], self.cropsz)

        return outputs


    def load(self, fpath):
        chkpt = torch.load(fpath)
        # Backward compatibility
        state_dict = chkpt['state_dict'] if 'state_dict' in chkpt else chkpt
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)
