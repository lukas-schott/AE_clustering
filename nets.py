from lasagne import layers as L
from theano import tensor as T
import theano
import lasagne as las


def AE_v0():
    layers = {}
    units = [20 * 10, 2000, 500, 2]
    layers['l_in_raw'] = L.InputLayer((None, units[0]))
    # encoder
    layers['en_00'] = L.DenseLayer(layers['l_in_raw'], units[0], 
                                   nonlinearity=las.nonlinearities.tanh)
    layers['en_01'] = L.DenseLayer(layers['en_00'], units[1], 
                                   nonlinearity=las.nonlinearities.tanh)
    layers['en_02'] = L.DenseLayer(layers['en_01'], units[2], 
                                   nonlinearity=las.nonlinearities.tanh)
    layers['en_03'] = L.DenseLayer(layers['en_02'], units[3], 
                                   nonlinearity=las.nonlinearities.tanh)

    # decoder
    layers['de_00'] = L.DenseLayer(layers['en_03'], units[2],
                                   nonlinearity=las.nonlinearities.tanh)
    layers['de_01'] = L.DenseLayer(layers['de_00'], units[1],
                                   nonlinearity=las.nonlinearities.tanh)
    layers['de_02'] = L.DenseLayer(layers['de_01'], units[0],
                                   nonlinearity=las.nonlinearities.tanh)
   #layers['de_03'] = L.DenseLayer(layers['de_02'], units[0],nonlinearity=None)
   # why do we need de_03?
    return layers

def net_fct(layers,LR):

    functions = {}

    # loass function
    all_params = L.get_all_params(layers['de_02'])
    l_out = L.get_output(layers['de_02'])
    l_in = L.get_output(layers['l_in_raw'])
    L1 = las.regularization.regularize_layer_params(layers['de_02'], 
                                                    las.regularization.l1)
    loss = T.mean((l_out - l_in)**2) + L1 * 10 ** - 6

    updates = las.updates.nesterov_momentum(loss, all_params, LR)
    functions['loss_f'] = theano.function([layers['l_in_raw'].input_var], 
                                          loss, updates=updates)

    # clustering
    l_middle = L.get_output(layers['en_03'])
    functions['middle_f'] = theano.function([layers['l_in_raw'].input_var], 
                                            l_middle)

    # outpu

    l_middle = L.get_output(layers['de_02'])
    functions['out_f'] = theano.function([layers['l_in_raw'].input_var], 
                                         l_middle)
    return functions




if __name__ == '__main__':
    layers = AE_v0()
    net_fct(layers)
