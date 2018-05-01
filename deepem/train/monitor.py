from __future__ import print_function
from collections import OrderedDict
import os
import h5py


class LearningMonitor:
    """
    LearningMonitor - a record keeping class for training
    neural nets, including functionality for maintaining
    running averages.
    """
    def __init__(self, fname=None):
        if fname is None:
            # Each dict holds nums & denoms for each running average it records.
            self.train = dict(numerators=dict(), denominators=dict())
            self.test  = dict(numerators=dict(), denominators=dict())
        else:
            self.load(fname)

    def append_train(self, iter, data):
        self._append(iter, data, 'train')

    def append_test(self, iter, data):
        self._append(iter, data, 'test')

    def add_to_num(self, data, phase):
        self._add_to_avg(data, True, phase)

    def add_to_denom(self, data, phase):
        self._add_to_avg(data, False, phase)

    def get_last_iter(self):
        """Return the last iteration number."""
        ret = 0
        if 'iter' in self.train and 'iter' in self.test:
            ret = max(self.train['iter'][-1],self.test['iter'][-1])
        return ret

    def get_last_value(self, key, phase):
        """Extract the last value from one of the records."""
        assert phase in ['train','test'], "invalid phase {}".format(phase)
        d = getattr(self, phase)
        return d[key][-1]

    def get_last_values(self, phase):
        """Extract the last values."""
        assert phase in ['train','test'], "invalid phase {}".format(phase)
        d = getattr(self, phase)
        return {k: d[k][-1] for k in d}

    def load(self, fname):
        """Initialize by loading from a h5 file."""
        assert(os.path.exists(fname))
        f = h5py.File(fname, 'r', driver='core')
        # Train stats.
        train = f['/train']
        for key, data in train.items():
            self.train[key] = list(data.value)
        # Test stats.
        test = f['/test']
        for key, data in test.items():
            self.test[key] = list(data.value)
        f.close()

    def save(self, fname, elapsed, base_lr=0):
        """Save stats."""
        if os.path.exists(fname):
            os.remove(fname)
        # Crate h5 file to save.
        f = h5py.File(fname)
        # Train stats.
        for key, data in self.train.items():
            if key in ['numerators','denominators']:
                continue
            f.create_dataset('/train/{}'.format(key), data=data)
        # Test stats.
        for key, data in self.test.items():
            if key in ['numerators','denominators']:
                continue
            f.create_dataset('/test/{}'.format(key), data=data)
        # Iteration speed in (s/iteration).
        f.create_dataset('/elapsed', data=elapsed)
        f.create_dataset('/base_lr', data=base_lr)
        f.close()

    def compute_avgs(self, iter, phase):
        """
        Finalizes the running averages, and appends them
        onto the train & test records.
        """
        d = getattr(self, phase)
        nums = d['numerators']
        denoms = d['denominators']

        avgs = {k: nums[k]/denoms[k] for k in nums.keys()}
        self._append(iter, avgs, phase)

        # Resetting averages.
        for k in nums.keys():
            nums[k], denoms[k] = 0.0, 0.0

    ####################################################################
    ## Non-interface functions.
    ####################################################################

    def _add_to_avg(self, data, numerators, phase):
        assert phase in ['train','test'], "invalid phase {}".format(phase)
        term = 'numerators' if numerators else 'denominators'
        d = getattr(self, phase)[term]
        for key, val in data.items():
            if key not in d:
                d[key] = 0.0
            d[key] += val

    def _append(self, iter, data, phase):
        assert phase in ['train','test'], "invalid phase {}".format(phase)
        d = getattr(self, phase)
        # Iteration.
        if 'iter' not in d:
            d['iter'] = list()
        d['iter'].append(iter)
        # Stats.
        for key, val in data.items():
            if key not in d:
                d[key] = list()
            d[key].append(val)
