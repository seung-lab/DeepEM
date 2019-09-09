from __future__ import print_function
import numpy as np
import time

import torch

from dataprovider3 import Dataset, ForwardScanner

from deepem.test import fwd_utils


class Forward(object):
    """
    Forward scanning.
    """
    def __init__(self, opt):
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.scan_spec = dict(opt.scan_spec)
        self.scan_params = dict(opt.scan_params)
        self.test_aug = opt.test_aug
        self.variance = opt.variance
        self.precomputed = (opt.blend == 'precomputed')

    def __call__(self, model, scanner):
        dataset = scanner.dataset

        # Test-time augmentation
        if self.test_aug:

            # For variance computation
            if self.variance:
                aug_out = dict()
                for k, v in scanner.outputs.data.items():
                    aug_out[k] = list()
            else:
                aug_out = None

            count = 0.0
            for aug in self.test_aug:
                # dec2bin
                rule = np.array([int(x) for x in bin(aug)[2:].zfill(4)])
                print("Test-time augmentation {}".format(rule))

                # Augment dataset.
                aug_dset = Dataset(spec=self.in_spec)
                for k, v in dataset.data.items():
                    aug_dset.add_data(k, fwd_utils.flip(v._data, rule=rule))

                # Forward scan
                aug_scanner = self.make_forward_scanner(aug_dset)
                outputs = self.forward(model, aug_scanner)

                # Accumulate.
                for k, v in scanner.outputs.data.items():
                    print("Accumulate to {}...".format(k))
                    output = outputs.get_data(k)

                    # Revert output.
                    dst = (1,1,1) if k == 'affinity' else None
                    reverted = fwd_utils.revert_flip(output, rule=rule, dst=dst)
                    v._data += reverted

                    # For variance computation
                    if self.variance:
                        aug_out[k].append(reverted)

                count += 1

            # Normalize.
            for k, v in scanner.outputs.data.items():
                print("Normalize {}...".format(k))
                if self.precomputed:
                    v._data[...] /= count
                else:
                    v._norm._data[...] = count

            return (scanner.outputs, aug_out)

        return (self.forward(model, scanner), None)

    ####################################################################
    ## Non-interface functions
    ####################################################################

    def forward(self, model, scanner):
        elapsed = list()
        t0 = time.time()
        with torch.no_grad():
            inputs = scanner.pull()
            while inputs:
                inputs = self.to_torch(inputs)

                # Forward pass
                outputs = model(inputs)
                scanner.push(self.from_torch(outputs))

                # Elapsed time
                elapsed.append(time.time() - t0)
                print("Elapsed: %.3f s" % elapsed[-1])
                t0 = time.time()

                # Fetch next inputs
                inputs = scanner.pull()

        print("Elapsed: %.3f s/patch" % (sum(elapsed)/len(elapsed)))
        print("Throughput: %d voxel/s" % round(scanner.voxels()/sum(elapsed)))
        return scanner.outputs

    def to_torch(self, sample):
        inputs = dict()
        for k in sorted(self.in_spec):
            data = np.expand_dims(sample[k], axis=0)
            tensor = torch.from_numpy(data)
            inputs[k] = tensor.cuda()
        return inputs

    def from_torch(self, outputs):
        ret = dict()
        for k in sorted(self.out_spec):
            if k in self.scan_spec:
                scan_channels = self.scan_spec[k][-4]
                narrowed = outputs[k].narrow(1, 0, scan_channels)
                ret[k] = np.squeeze(narrowed.cpu().numpy(), axis=(0,))
        return ret

    def make_forward_scanner(self, dataset):
        return ForwardScanner(dataset, self.scan_spec, **self.scan_params)
