import nets
from dataset_utils import BatchMan
import theano
from theano.sandbox import cuda as s
from matplotlib import pyplot as plt
import progressbar
import numpy as np
from scipy import ndimage

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

        if iteration % 5000 == 0:
            bm.bs = 100000 - 10-1
            batch, spikes = bm.get_batch(sorted=True)
            middle = functions['middle_f'](batch)

            xs = middle[:, 0]
            ys = middle[:, 1]

            # ys = np.ones_like(xs)
            fig, ax = plt.subplots(2)
            ax[0].clear()
            ax[1].clear()
            ax[0].scatter(xs, ys, alpha=0.3, c=spikes, marker=spikes)
            # ax[0].scatter(xs[spikes == 0], ys[spikes == 0],  marker='^', alpha=0.3, c=(0, 0, 0))
            # ax[0].scatter(xs[spikes == 1], ys[spikes == 1],  marker='8',alpha=0.3, c=(0, 0.5, 0))
            # ax[0].scatter(xs[spikes == 2], ys[spikes == 2], marker='p',  alpha=0.3, c=(1, 0, 0))


            xsm = ndimage.convolve(xs, np.ones(10) / 10., mode='mirror')
            ysm = ndimage.convolve(ys, np.ones(10) / 10., mode='mirror')

            ax[1].scatter(xsm, ysm,  c=spikes, alpha=0.3)
            # ax[1].scatter(xsm[spikes == 0], ysm[spikes == 0],  marker='^', alpha=0.3, c=(0, 0, 0))
            # ax[1].scatter(xsm[spikes == 1], ysm[spikes == 1],  marker='8', alpha=0.3, c=(0, 0.5, 0))
            # ax[1].scatter(xsm[spikes == 2], ysm[spikes == 2], marker='p', alpha=0.3, c=(1, 0, 0))
            fig.savefig('./../data/plots2/tDAllNormalplot_%i.png' % iteration)
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