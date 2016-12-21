import nets
from dataset_utils import BatchMan
import theano
from theano.sandbox import cuda as s
from matplotlib import pyplot as plt
import progressbar
import numpy as np

def train_net():
    n_iter = 500000
    bs = 10
    layers = nets.AE_v0()
    functions = nets.net_fct(layers)
    bm = BatchMan(bs=bs)

    bar = progressbar.ProgressBar(max_value=n_iter)

    batch, _ = bm.get_batch()
    middle = functions['middle_f'](batch)
    out = functions['out_f'](batch)

    for iteration in range(n_iter):
        batch, _ = bm.get_batch()
        # print 'middle', middle
        loss = functions['loss_f'](batch)

        if iteration % 10:
            bar.update(iteration)
        if iteration % 1000 == 0:
            print '\r loss %.3f' % loss

        if iteration % 1000 == 0:
            bm.bs = 100000 - 10-1
            batch, spikes = bm.get_batch()
            middle = functions['middle_f'](batch)

            xs = middle[:, 0]
            ys = middle[:, 1]
            # ys = np.ones_like(xs)
            fig, ax = plt.subplots()
            ax.clear()
            ax.scatter(xs, ys, c=spikes,  alpha=0.3)
            fig.savefig('./../data/plots/2DAllNormalplot_%i.png' % iteration)
            plt.close(fig)
            bm.bs = bs


    bm.bs = 10000
    batch = bm.get_batch()
    middle = functions['middle_f'](batch)

    xs = middle[:, 0]
    ys = middle[:, 1]

    plt.scatter(xs, ys, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    s.use('gpu0')
    train_net()