import h5py as h5
import numpy as np


class BatchMan():
    def __init__(self, bs=10, sequ_len=10):
        self.data = self.load_data()
        self.T_max = self.data.shape[-1]
        self.n_neurons = self.data.shape[0]
        self.batch_sequ_len = sequ_len
        self.bs = bs

    def load_data(self):
        path = './../data/AE_data.h5'
        with h5.File(path, 'r') as f:
            data = np.array(f['data'])
        return data

    def get_batch(self):
        input_batch = np.empty((self.bs, self.batch_sequ_len * self.n_neurons), dtype=np.float32)
        t_selections = np.random.choice(100000 - 10 - 1, size=self.bs, replace=False)
        for b in range(self.bs):
            input_batch[b, :] = self.data[:, t_selections[b]:t_selections[b] + self.batch_sequ_len].flatten()
        return input_batch


if __name__ == '__main__':
    bm = BatchMan()
    bm.get_batch()
    print 'ba'