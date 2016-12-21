import h5py as h5
import numpy as np


class BatchMan():
    def __init__(self, bs=10, sequ_len=10):
        self.data, self.spikes = self.load_data()
        self.T_max = self.data.shape[-1]
        self.n_neurons = self.data.shape[0]
        self.batch_sequ_len = sequ_len
        self.bs = bs
        self.labels = np.zeros(len(self.spikes[0]))

        self.detect_classes()

    def detect_classes(self):

        self.labels[self.spikes[0] == 1] = 1
        self.labels[self.spikes[1] == 1] = 2
        print 'lalbels 0', self.labels


    def load_data(self):
        # path = './../data/AE_data.h5'
        path = './../data/AE_data_more_spkes.h5'
        with h5.File(path, 'r') as f:
            data = np.array(f['data'])
            spikes = np.array(f['spikes'])
        return data, spikes

    def get_batch(self, sorted=False):
        input_batch = np.empty((self.bs, self.batch_sequ_len * self.n_neurons), dtype=np.float32)
        if not sorted:
            t_selections = np.random.choice(self.T_max - self.batch_sequ_len - 1, size=self.bs, replace=False)
        else:
            t_selections = np.arange(self.T_max - self.batch_sequ_len - 1)

        for b in range(self.bs):
            input_batch[b, :] = self.data[:, t_selections[b]:t_selections[b] + self.batch_sequ_len].flatten()
        return input_batch, self.labels[t_selections]


if __name__ == '__main__':
    bm = BatchMan()
    bm.detect_classes()
    bm.get_batch()
    print 'ba'