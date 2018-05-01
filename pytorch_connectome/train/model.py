from __future__ import print_function


import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()

        self.pretrain = opt.pretrain

    def forward(self, sample):
        """Forward pass & loss."""
        # Forward pass
        inputs = [sample[k] for k in sorted(self.in_spec)]
        outputs = self.model(*inputs)
        # Evaluate loss.
        losses, nmasks = self.eval_loss(outputs, sample)
        labels = [sample[k] for k in sorted(self.out_spec)]
        return losses, nmasks, inputs, preds, labels

    def eval_loss(self, preds, sample):
        loss = OrderedDict()
        nmsk = OrderedDict()
        for i, k in enumerate(sorted(self.out_spec)):
            label = sample[k]
            mask = sample[k+'_mask']
            if k == 'affinity':
                loss[k], nmsk[k] = self.loss_fn(preds[i], label, mask)
            else:
                loss[k], nmsk[k] = self.bceloss(preds[i], label, mask)
        return list(loss.values()), list(nmsk.values())

    def save(self, fpath):
        torch.save(self.model.state_dict(), fpath)

    def load(self, fpath):
        state_dict = torch.load(fpath)
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)
