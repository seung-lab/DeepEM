from __future__ import print_function
import numpy as np
import time

import torch


class Forward(object):
    """
    Forward scanning.
    """
    def __init__(self, opt):
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.scan_spec = dict(opt.scan_spec)

    def __call__(self, model, scanner):
        return self.forward(model, scanner)

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
                outputs = model(*inputs)
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
        inputs = list()
        for k in sorted(self.in_spec):
            data = np.expand_dims(sample[k], axis=0)
            tensor = torch.from_numpy(data).cuda()
            inputs.append(tensor.cuda())
        return inputs

    def from_torch(self, outputs):
        ret = dict()
        for k in sorted(self.out_spec):
            if k in self.scan_spec:
                scan_channels = self.scan_spec[k][-4]
                narrowed = outputs[k].narrow(1, 0, scan_channels)
                ret[k] = np.squeeze(narrowed.cpu().numpy(), axis=(0,))
        return ret
