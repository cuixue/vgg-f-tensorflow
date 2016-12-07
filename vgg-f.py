#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np
import scipy.misc
import os
import scipy.io
class Vggf:
    def __init__(self, data_path):
        data = scipy.io.loadmat(data_path)
        layers = (
            'conv1', 'relu1', 'norm1', 'pool1',

            'conv2', 'relu2', 'norm2', 'pool2',

            'conv3', 'relu3', 'conv4', 'relu4',
            'conv5', 'relu5', 'pool5',
            'fc6', 'relu6', 'fc7', 'relu7'

        )
        weights = data['layers'][0]
        mean = data['normalization']
        net = {}

        self.current = tf.placeholder('float32',[None,224,224,3])
        current = self.current
        for i, name in enumerate(layers):
            if name.startswith('conv'):
                kernels, bias = weights[i][0][0][0][0]
                bias = bias.reshape(-1)
                pad = weights[i][0][0][1]
                stride = weights[i][0][0][4]
                current = _conv_layer(current, kernels, bias, pad, stride, i,  net)
            elif name.startswith('relu'):
                current = tf.nn.relu(current)
            elif name.startswith('pool'):
                stride = weights[i][0][0][1]
                pad = weights[i][0][0][2]
                area = weights[i][0][0][5]
                current = _pool_layer(current, stride, pad, area)
            elif name.startswith('fc'):
                kernels, bias = weights[i][0][0][0][0]
                bias = bias.reshape(-1)
                current = _full_conv(current, kernels, bias, i, net)
            elif name.startswith('norm'):
                current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            net[name] = current
        self.net = net
        self.mean = mean
        #return net, mean, ops


def _conv_layer(input, weights, bias, pad, stride, i, net):
    pad = pad[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[1, stride[0], stride[1], 1], padding='VALID', name='conv' + str(i))
    return tf.nn.bias_add(conv, b, name='add' + str(i))


def _full_conv(input, weights, bias, i, net):
    w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID', name='fc' + str(i))
    return tf.nn.bias_add(conv, b, name='add' + str(i))


def _pool_layer(input, stride, pad, area):
    pad = pad[0]
    area = area[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1], padding='VALID')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def get_meanpix(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    return mean

def main():
    data_path = './cifar-10-batches-mat/imagenet-vgg-f.mat'
    data = np.ones([10,224,224,3])

    gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4))
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    with tf.Graph().as_default(), tf.Session(config=gpuconfig) as session:
        model = Vggf(data_path)
        session.run(tf.initialize_all_variables())
        print tf.reduce_sum(model.net['relu7']).eval(feed_dict={model.current: data})

if __name__ == '__main__':
        main()