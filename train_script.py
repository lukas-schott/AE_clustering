import nets
from dataset_utils import BatchMan
import theano
from theano.sandbox import cuda as s
from matplotlib import pyplot as plt
import progressbar

def train_net():
    n_iter = 500000
    bs = 10
    layers = nets.AE_v0()
    functions = nets.net_fct(layers)
    bm = BatchMan(bs=bs)

    bar = progressbar.ProgressBar(max_value=n_iter)

    batch = bm.get_batch()
    middle = functions['middle_f'](batch)
    out = functions['out_f'](batch)

    for iteration in range(n_iter):
        batch = bm.get_batch()
        # print 'middle', middle
        loss = functions['loss_f'](batch)

        bar.update(iteration)
        if iteration % 100 == 0:
            print '\r loss %.3f' % loss

        if iteration % 10000 == 0:
            bm.bs = 50000
            batch = bm.get_batch()
            middle = functions['middle_f'](batch)

            xs = middle[:, 0]
            ys = middle[:, 1]
            fig = plt.figure()
            plt.scatter(xs, ys, alpha=0.3)
            plt.savefig('./../data/Normalplot_%i.png' % iteration)
            bm.bs = bs


    bm.bs = 10000
    batch = bm.get_batch()
    middle = functions['middle_f'](batch)

    xs = middle[:, 0]
    ys = middle[:, 1]

    plt.scatter(xs, ys, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    # s.use('gpu0')
    train_net()