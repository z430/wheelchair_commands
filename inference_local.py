import numpy as np
import speechpy as spy
import librosa
from deepnet.layers import *
from deepnet.nnet import CNN
import h5py
import audio_utility as au
import scipy
import conv2d
import mxnet as mx
from mxnet import nd

label_index = ['_silence_', '_unknown_', 'down', 'go', 'left', 'no', 'off',
               'on', 'right', 'stop', 'up', 'yes']
label_index = ['_silence_', '_unknown_', 'marvin']


def get_weight(key):
    hf = h5py.File('data/ex_conv_mfcc_psf_marvin2_.h5', 'r')
    return np.array(hf[key])


def dnn_arch(input_shape, n_class):
    fc1 = FullyConnected(input_shape[0], 144, weight=get_weight('layer_1weight').T,
                         bias=get_weight('layer_1bias'))
    sig1 = ReLU()
    fc2 = FullyConnected(144, 144, weight=get_weight('layer_2weight').T,
                         bias=get_weight('layer_2bias'))
    sig2 = ReLU()
    fc3 = FullyConnected(144, 144, weight=get_weight('layer_3weight').T,
                         bias=get_weight('layer_3bias'))
    sig3 = ReLU()
    fc4 = FullyConnected(144, 144, weight=get_weight('layer_4weight').T,
                         bias=get_weight('layer_4bias'))
    sig4 = ReLU()
    fc5 = FullyConnected(144, n_class, weight=get_weight('layer_5weight').T,
                         bias=get_weight('layer_5bias'))
    return [fc1, sig1, fc2, sig2, fc3, sig3, fc4, sig4, fc5]


def cnn_arch(input_shape, n_class):
    conv = Conv(input_shape, n_filter=64, h_filter=20,
                w_filter=8, stride=1, padding=0, weight=get_weight('layer_1a').T,
                bias=get_weight('layer_1b'))
    # print("Conv1 shape: ", conv.out_dim)
    relu = ReLU()
    dropout = Dropout()
    # maxpool = Maxpool(conv.out_dim, 2, 1)
    # print("Maxpool shape: ", maxpool.out_dim)
    conv2 = Conv(conv.out_dim, n_filter=32, h_filter=10,
                 w_filter=4, stride=1, padding=0, weight=get_weight('layer_3a').T,
                 bias=get_weight('layer_3b'))
    relu2 = ReLU()
    dropout2 = Dropout()

    flat = Flatten()
    # print("Conv2 shape: ", conv2.out_dim)
    fc = FullyConnected(np.prod(conv2.out_dim), n_class, weight=get_weight('layer_6a').T,
                        bias=get_weight('layer_6b'))
    return [conv, relu, dropout, conv2, relu2, dropout2, flat, fc]


def conv2d_(input, n_filter, h_filter, w_filter, stride, padding, weight, bias):
    d_X, h_X, w_X = input.shape

    bias = bias[:, np.newaxis]

    h_out = (h_X - h_filter + 2 * padding) / stride + 1
    w_out = (w_X - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    out_dim = (n_filter, h_out, w_out)
    print("CONV output_dim : ", out_dim)

    input = input[np.newaxis, :, :, :]
    print("input_shapes: ", input.shape)
    n_X = input.shape[0]
    X_col = im2col_indices(
        input, h_filter, w_filter, stride=stride, padding=padding
    )
    print("weigh shape: ", weight.shape)
    W_row = weight.reshape(n_filter, -1)
    print("X_Col conv2d: ", X_col.shape, W_row.shape)
    out = W_row @ X_col + bias
    out = out.reshape(n_filter, h_out, w_out, n_X)
    out = out.transpose(3, 0, 1, 2)
    out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
    print("Conv2d output: ", out.shape)
    return out


def convolve2d(x, nf, rf, s, b, np_o, w):
    """

    :param x: input
    :param nf: number of filter
    :param rf: filter size
    :param s: stride
    :param b: bias
    :param np_o: numpy outpur
    :param w: weight
    :return:
    """
    for z in range(nf):
        print("z:", z)
        h_range = int((x.shape[2] - rf) / s) + 1  # (W - F + 2P) / S
        for _h in range(h_range):
            w_range = int((x.shape[1] - rf) / s) + 1  # (W - F + 2P) / S
            for _w in range(w_range):
                np_o[0, _h, _w, z] = np.sum(
                    x[0, _h * s:_h * s + rf, _w * s:_w * s + rf, :] *
                    w[:, :, :, z]) + b[z]


def pooling(feature_map):
    # feature_map = np.load('conv1_out.npy')
    feature_map = np.reshape(feature_map, (26, 99, 64))
    print("feat shape: ", feature_map.shape)
    size = 1
    stride = 2
    # Preparing the output of the pooling operation.
    pool_out = np.zeros((np.uint16((feature_map.shape[0] - size + 1) / stride),
                         np.uint16((feature_map.shape[1] - size + 1) / stride),
                         feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0] - size - 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size - 1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r + size, c:c + size]])
                c2 = c2 + 1
            r2 = r2 + 1
    # print(pool_out)
    return pool_out


def flatten(X):
    X_shape = X.shape
    out_shape = (X_shape[0], -1)
    print(X.shape)
    out = X.ravel().reshape(out_shape)
    out_shape = out_shape[1]
    return out


def relu(X):
    return np.maximum(X, 0)


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z)))
    return sm


def conv2(y_mfcc):
    cnn_weight_1 = get_weight('Variable_0').T
    cnn_bias_1 = get_weight('Variable_1_0')

    print("weight1: ", cnn_weight_1.shape)

    # convolve2d(y_mfcc, 64, 20, 1, cnn_bias_1)
    c1 = conv2d.conv2d(y_mfcc, cnn_weight_1) + cnn_bias_1
    # c1 = c1.transpose(0, 2, 1, 3) + cnn_bias_1
    c1 = relu(c1)
    c6 = pooling(c1)
    print("pooling result: ", c6.shape)
    print(c1.shape)
    import skimage.measure
    c2 = skimage.measure.block_reduce(c1, (1, 2, 2, 1), np.mean)
    print(c2.shape)

    cnn_weight_2 = get_weight('Variable_2_0').T
    cnn_bias_2 = get_weight('Variable_3_0')
    print("weight2: ", cnn_weight_2.shape)
    c3 = conv2d.conv2d(c2, cnn_weight_2) + cnn_bias_2
    c3 = relu(c3)
    print(c3.shape)
    el_count = c3.shape[0] * c3.shape[1] * c3.shape[2] * c3.shape[3]
    c4 = np.reshape(c3, (-1, el_count))
    print("c4: ", c4.shape)

    fp_weight = get_weight('Variable_4_0').T
    fp_bias = get_weight('Variable_5_0')
    c5 = c4.dot(fp_weight)
    c5 = c5 + fp_bias
    print(c5)
    print(label_index[np.argmax(c5)])
    c5 = softmax(c5)
    print(c5.flatten())
    print(label_index[np.argmax(c5)])


def relu2(X):
    return nd.maximum(X, nd.zeros_like(X))


def conv3(y_mfcc):
    y_mfcc = nd.array(y_mfcc)
    W1 = nd.array(get_weight('layer_1a'))
    # print(W1.shape)
    W1 = nd.transpose(W1, (3, 2, 0, 1))

    b1 = nd.array(get_weight('layer_1b'))
    # ctx = mx.cpu()

    W2 = get_weight('layer_4a')
    b2 = get_weight('layer_4b')

    W3 = get_weight('layer_7a')
    b3 = get_weight('layer_7b')

    conv1 = nd.Convolution(y_mfcc, weight=W1, bias=b1, kernel=(20, 8), num_filter=64)
    conv1 = relu2(conv1)
    pool1 = nd.Pooling(conv1, pool_type="avg", kernel=(3, 1), stride=(3, 1))
    print(pool1.shape)
    conv2 = nd.Convolution(pool1, nd.array(np.transpose(W2, (3, 2, 0, 1))), bias=nd.array(b2), kernel=(10, 4),
                           num_filter=64)
    print(conv2.shape)
    conv2 = relu2(conv2)
    fla1 = nd.flatten(conv2)

    # fp1 = nd.dot(fla1, nd.array(W3)) + nd.array(b3)
    print(conv1.shape, pool1.shape, conv2.shape, fla1.shape,
          # fp1.shape
          )
    # labels = np.argmax(np.array(fp1))
    # output = (fp1.asnumpy()).flatten()
    # # print(fp1.asnumpy())
    # print(softmax(output), labels, label_index[np.argmax(softmax(output))])
    # print(labels)
    # return output, label_index[labels]


def conv4(y_mfcc):
    input_shape = [1, 99, 26]
    cnn = CNN(cnn_arch(input_shape, 3))
    resu = cnn.predict(y_mfcc)
    # print(label_index[np.argmax(resu, axis=1)[0]])
    return resu, label_index[np.argmax(resu, axis=1)[0]]


def main():
    # read wav
    y, fs = librosa.load('../dataset/test_set/_unknown_/six_3efef882_nohash_0.wav', sr=16000)
    # mfcc
    y_mfcc = (au.mfcc_psf(y))
    y_mfcc = y_mfcc[np.newaxis, np.newaxis, :, :]
    # conv_forward_strides(y_mfcc, w=get_weight('layer_1a').T, b=get_weight('layer_1b'))

    print("input shape ", y_mfcc.shape)
    # conv2(y_mfcc)
    print(conv4(y_mfcc))

    # az = scipy.ndimage.filters.convolve(y_mfcc, cnn_weight_1)
    # print(az.shape)

    # c1 = conv2d(y_mfcc, n_filter=64, h_filter=20, w_filter=8, stride=1, padding=0, weight=get_weight('layer_1a'),
    #             bias=get_weight('layer_1b'))
    # # conv_forward(y_mfcc, weight, bias)
    # c1 = relu(c1)
    # print("m1 output: ", c1.shape)
    # print("===============================================================================================")
    # c2 = conv2d(c1, n_filter=64, h_filter=10, w_filter=4, stride=1, padding=0, weight=get_weight('layer_3a'),
    #             bias=get_weight('layer_3b'))
    # c2 = relu(c2)
    # print(c2.shape)
    # fl = c2.flatten()
    # print(fl.shape)
    # weight2 = get_weight('layer_6a')
    # bias2 = get_weight('layer_6b')
    # out3 = fl.dot(weight2) + bias2
    # out3 = softmax(out3)
    # print(out3.shape)
    # print(np.argmax(out3))
    # print(label_index[np.argmax(out3)])
    # create cnn-> 28 10 4 1 1 30 10 4 2 1 16 128


if __name__ == '__main__':
    main()
