from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../modular_semantic_segmentation')

from experiments.utils import get_observer, load_data
from experiments.evaluation import evaluate, import_weights_into_network

from xview.datasets import Cityscapes_GAN
from xview.datasets import get_dataset
from xview.settings import EXP_OUT

import os
import sacred as sc
from sacred.utils import apply_backspaces_and_linefeeds
import scipy.misc
import numpy as np
import shutil
import tensorflow as tf
import argparse
import json
import glob
import random
import collections
import math
import time
import scipy
import cv2
from copy import deepcopy
from sys import stdout
from skimage.measure import compare_ssim
from model import pix2pix

Model = collections.namedtuple("Model", "inputs, targets, outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars,global_step, train")

class Helper:
    name = 'A'

a = Helper()

EPS = 1e-12
num_test_images = 20

def create_directories(run_id, experiment):
    """
    Make sure directories for storing diagnostics are created and clean.

    Args:
        run_id: ID of the current sacred run, you can get it from _run._id in a captured
            function.
        experiment: The sacred experiment object
    Returns:
        The path to the created output directory you can store your diagnostics to.
    """
    root = EXP_OUT
    # create temporary directory for output files
    if not os.path.exists(root):
        os.makedirs(root)
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '{}/{}'.format(root, run_id)
    if os.path.exists(output_dir):
        # Directory may already exist if run_id is None (in case of an unobserved
        # test-run)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    experiment.info.setdefault("tensorflow", {}).setdefault("logdirs", [])\
        .append(output_dir)
    return output_dir

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

def add_noise(image):
    with tf.name_scope("add_noise"):
        return image+tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=a.noise_std_dev, dtype=tf.float32)

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return ((image + 1) / 2)*255
def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(inputs=batch_input, filters=out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
def gen_deconv(batch_input, out_channels, image_size):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        # _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=a.batch_momentum, training=a.is_train, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(add_noise(generator_inputs), a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(add_noise(layers[-1]), 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    images_sizes = [
        int(a.input_image_size * 0.5**7),
        int(a.input_image_size * 0.5**6),
        int(a.input_image_size * 0.5**5),
        int(a.input_image_size * 0.5**4),
        int(a.input_image_size * 0.5**3),
        int(a.input_image_size * 0.5**2),
        int(a.input_image_size * 0.5),
        int(a.input_image_size)
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = add_noise(layers[-1])
            else:
                input = add_noise(tf.concat([layers[-1], layers[skip_layer]], axis=3))

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels,images_sizes[decoder_layer])
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = add_noise(tf.concat([layers[-1], layers[0]], axis=3))
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels,images_sizes[-1])
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def create_model(inp, tar):
    inputs = preprocess(inp)
    targets = preprocess(tar)
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        if a.label_smoothing==True:
            smoothing_fake = tf.random_uniform(shape=[1],minval=1,maxval=1.3)
            smoothing_real = tf.random_uniform(shape=[1],minval=0.0,maxval=0.2)
        else:
            smoothing_fake = 1
            smoothing_real = 0

        cond = tf.random_uniform(shape=[1],minval=0,maxval=1)
        discrim_loss = tf.cond(cond[0] > (a.flip_prob+a.flip_end_prob),
                               lambda: tf.reduce_mean(-(tf.log(smoothing_real + predict_real + EPS) + tf.log(smoothing_fake - predict_fake + EPS))),
                               lambda: tf.reduce_mean(-(tf.log(smoothing_real + predict_fake + EPS) + tf.log(smoothing_fake - predict_real + EPS))))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.name_scope("discriminator_train"):
        with tf.control_dependencies(update_ops):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    update_flip_prob = tf.cond(global_step > a.flip_decay_step,
                           lambda: tf.assign(a.flip_prob, a.flip_prob*a.decay_term),
                           lambda: a.flip_prob)

    return Model(
        inputs=inp,
        targets=tar,
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=deprocess(outputs),
        global_step=global_step,
        train=tf.group(update_losses, incr_global_step, gen_train, update_flip_prob),
    )

@ex.main
def main(dataset, net_config, _run):
    # Add all of the config into the helper class
    for key in net_config:
        setattr(a, key, net_config[key])

    output_dir = create_directories(_run._id, ex)
    # load the dataset class
    data = get_dataset(dataset['name'])
    data = data(**dataset)

    # Get the data descriptors with the shape of data coming
    data_description = data.get_data_description()
    data_description = [data_description[0], {
        key: [None, *description]
        for key, description in data_description[1].items()}]

    # Create an iterator for the data
    iter_handle = tf.placeholder(tf.string, shape=[],
                                          name='training_placeholder')
    iterator = tf.data.Iterator.from_string_handle(
        iter_handle, *data_description)
    training_batch = iterator.get_next()

    # Create pix2pix model
    model = create_model(training_batch['labels'], training_batch['rgb'])


    if a.mode=="train":
        print("INFO: Got train set")
        input_data = data.get_trainset()
        data_iterator = input_data.repeat(a.max_epochs).batch(a.batch_size).make_one_shot_iterator()
        # data_iterator = input_data.batch(a.batch_size).make_initializable_iterator()

        validation_data = data.get_validation_set()
        valid_iterator = validation_data.batch(a.batch_size).make_one_shot_iterator()
        valid_handle = sess.run(valid_iterator.string_handle())
    else:
        print("INFO: Got test set")
        input_data = data.get_testset(num_items=num_test_images)
        data_iterator = input_data.batch(a.batch_size).make_one_shot_iterator()

    data_handle = sess.run(data_iterator.string_handle())

    if a.checkpoint is not None:
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(os.path.join(EXP_OUT,str(a.checkpoint)))
        saver.restore(sess, checkpoint)

    with tf.Session() as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)

        if args.phase == 'train':
            model.train(args)
        else:
            model.test(args)





if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
