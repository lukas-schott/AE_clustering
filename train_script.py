import nets
from dataset_utils import BatchMan
import theano
from theano.sandbox import cuda as s
from matplotlib import pyplot as plt
import progressbar
import numpy as np
from scipy import ndimage
import argparse
import os

# set by argparse
PLOTSUBDIR = None # Default: save directly into plots/
LR = None # Default: 0.01

def train_net():
    n_iter = 500000
    bs = 10
    layers = nets.AE_v0()
    functions = nets.net_fct(layers,LR=LR)
    bm = BatchMan(bs=bs)

    bar = progressbar.ProgressBar(max_value=n_iter)

    batch, _ = bm.get_batch()
    middle = functions['middle_f'](batch)
    out = functions['out_f'](batch)
    loss = 0.0

    for iteration in range(n_iter):
        batch, _ = bm.get_batch()
        # print 'middle', middle
        loss += functions['loss_f'](batch)

        if iteration % 10:
            bar.update(iteration)
        if iteration % 1000 == 0:
            print 'loss %.3f' % loss
            loss = 0.0

        if iteration % 10000 == 0:
            bm.bs = 100000 - 10-1
            batch, spikes = bm.get_batch(sorted=True)
            middle = functions['middle_f'](batch)

            xs = middle[:, 0]
            ys = middle[:, 1]

            # ys = np.ones_like(xs)
            fig, ax = plt.subplots(2)
            ax[0].clear()
            ax[1].clear()
            ax[0].scatter(xs, ys, alpha=0.3, c=spikes) #, marker=spikes)
            # ax[0].scatter(xs[spikes == 0], ys[spikes == 0],  marker='^', alpha=0.3, c=(0, 0, 0))
            # ax[0].scatter(xs[spikes == 1], ys[spikes == 1],  marker='8',alpha=0.3, c=(0, 0.5, 0))
            # ax[0].scatter(xs[spikes == 2], ys[spikes == 2], marker='p',  alpha=0.3, c=(1, 0, 0))


            xsm = ndimage.convolve(xs, np.ones(10) / 10., mode='mirror')
            ysm = ndimage.convolve(ys, np.ones(10) / 10., mode='mirror')

            ax[1].scatter(xsm, ysm,  c=spikes, alpha=0.3)
            # ax[1].scatter(xsm[spikes == 0], ysm[spikes == 0],  marker='^', alpha=0.3, c=(0, 0, 0))
            # ax[1].scatter(xsm[spikes == 1], ysm[spikes == 1],  marker='8', alpha=0.3, c=(0, 0.5, 0))
            # ax[1].scatter(xsm[spikes == 2], ysm[spikes == 2], marker='p', alpha=0.3, c=(1, 0, 0))
            fig.savefig('./../data/plots/%s/tDAllNormalPlot_%i.png'\
                        % (PLOTSUBDIR, iteration))
            plt.close(fig)

            # shift values for patterns before/after actual pattern
            sym_shifts = 7
            shifted_spikes = np.concatenate((np.zeros(sym_shifts),
                                             spikes,np.zeros(sym_shifts)))
            no_shifts = sym_shifts*2+1
            fig, axes = plt.subplots(3,5,figsize=(15,10))
            for shift in range(no_shifts):
              ax = axes.flatten()[shift]
              ax.clear()
              ax.set_title('Shift {}'.format(shift-sym_shifts))
              for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(10)
              active_spikes = shifted_spikes[shift:
                                             shift+spikes.shape[0]]
              ax.scatter(xs, ys, alpha=0.3, c=active_spikes)
            fig.tight_layout()
            fig.savefig('./../data/plots/%s/2DAllShiftPlot_%i.png'\
                        % (PLOTSUBDIR, iteration))
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

    parser = argparse.ArgumentParser(
                description='Train an Auto-Encoder on neuron '+
                            'spike patterns.')
    parser.add_argument("--plotsubdir", 
                        help="Save plots to subdirectory of plots/",
                        type=str, default="")
    parser.add_argument("--LR", 
                        help="learning rate",
                        type=float, default="0.01")
    args = parser.parse_args()

    PLOTSUBDIR = args.plotsubdir
    LR = args.LR
    if not os.path.isdir('./../data/plots/%s/' % PLOTSUBDIR):
      os.mkdir('./../data/plots/%s/' % PLOTSUBDIR)

    train_net()
