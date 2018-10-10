from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/Users/David/masterThesis/modular_semantic_segmentation')

from experiments.utils import get_observer, load_data
from experiments.evaluation import evaluate, import_weights_into_network

from xview.datasets import Cityscapes
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

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

#parser = argparse.ArgumentParser()
#a, unknown = parser.parse_args()
class Helper:
    name = 'A'

a = Helper()

ca = np.array([[0, 0, 0], #void
               [70,130,180], #sky
               [70, 70, 70], #building
               [128, 64, 128], #road
               [244, 35,232], #sidewalk
               [190,153,153], #fence
               [107,142, 35], #vegetation
               [153,153,153], #pole
               [0,  0, 142], #vehicle
               [220, 220, 0], #traffic sign
               [220, 20, 60], #person
               [119, 11, 32]]) #bicycle

EPS = 1e-12
CROP_SIZE = 256

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


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels,a):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples(data):
    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r


    if a.mode == "test":
        images = data.get_testset(tf_dataset=False)
        inputs = images['rgb']
        targets = ca[images['labels']]
    else:
        images = data.get_trainset(tf_dataset=False)
        inputs = images['rgb']
        targets = ca[images['labels']]


    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    inputs_batch, targets_batch = tf.train.batch([input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(np.shape(inputs)[0] / a.batch_size))

    return Examples(
        paths='',
        inputs=inputs_batch[0,...],
        targets=targets_batch[0,...],
        count=np.shape(inputs)[0],
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
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
            rectified = lrelu(layers[-1], 0.2)
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

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels,a)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels,a)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        discrim_targets = tf.cast(discrim_targets,tf.float32)
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
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        targets = tf.cast(targets,tf.float32)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
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

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, num_images, step=None):
    image_dir = os.path.join(a.file_output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # filesets = []
    # for i, in_path in enumerate(fetches["paths"]):
    #     name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
    #     fileset = {"name": name, "step": step}
    #     for kind in ["inputs", "outputs", "targets"]:
    #         filename = name + "-" + kind + ".png"
    #         if step is not None:
    #             filename = "%08d-%s" % (step, filename)
    #         fileset[kind] = filename
    #         out_path = os.path.join(image_dir, filename)
    #         contents = fetches[kind][i]
    #         with open(out_path, "wb") as f:
    #             f.write(contents)
    #     filesets.append(fileset)
    # return filesets

    filesets = []
    for i in range(num_images):
        name = "output" + str(i)
        fileset = {"name": name, "step": step}
        # for kind in ["outputs"]:
        kind ="outputs"
        filename = name + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][i]
        print(out_path)
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.file_output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

@ex.main
def main(dataset, net_config, _run):
# def main(modelname, dataset, net_config, _run):
    # Set up the directories for diagnostics

    for key in net_config:
        setattr(a, key, net_config[key])

    output_dir = create_directories(_run._id, ex)
    # load the dataset class, but don't instantiate it
    data = get_dataset(dataset['name'])
    data = data(**dataset)

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if a.mode == "test" or a.mode == "export":
        if a.file_output_dir is None:
            raise Exception("define a path to output folder")
        temp = a.file_output_dir+str(_run._id)
        setattr(a, "file_output_dir", temp)
        if not os.path.exists(a.file_output_dir):
             os.makedirs(a.file_output_dir)
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # examples = load_examples(data)

    desc, data_shape, num_classes = data.get_data_description()
    print(desc)
    print(data_shape)

    if a.mode == "test":
        samples = data.get_testset()
    else:
        samples = data.get_trainset()

    ###########################################################################
    #                       Try to load data as tf data                       #
    ###########################################################################

    # initialize datasets
    def _onehot_mapper(blob):
        blob['labels'] = tf.one_hot(blob['labels'], num_classes,
                                    dtype=tf.int32)
        return blob

    train_iterator = samples.map(_onehot_mapper, 10)\
        .repeat()\
        .batch(a.batch_size)\
        .make_one_shot_iterator()

    logdir = output_dir if ( a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        train_handle = sess.run(train_iterator.string_handle())


    ###########################################################################
    print("done with data loading")
    # print("examples count = %d" % examples.count)



    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    # if a.lab_colorization:
    #     if a.which_direction == "AtoB":
    #         # inputs is brightness, this will be handled fine as a grayscale image
    #         # need to augment targets and outputs with brightness
    #         targets = augment(examples.targets, examples.inputs)
    #         outputs = augment(model.outputs, examples.inputs)
    #         # inputs can be deprocessed normally and handled as if they are single channel
    #         # grayscale images
    #         inputs = deprocess(examples.inputs)
    #     elif a.which_direction == "BtoA":
    #         # inputs will be color channels only, get brightness from targets
    #         print(examples.inputs)
    #         print(examples.targets)
    #         inputs = augment(examples.inputs, examples.targets)
    #         targets = deprocess(examples.targets)
    #         outputs = deprocess(model.outputs)
    #     else:
    #         raise Exception("invalid direction")
    # else:
    # inputs = deprocess(examples.inputs)
    # targets = deprocess(examples.targets)
    # outputs = deprocess(model.outputs)
    #
    # def convert(image):
    #     if a.aspect_ratio != 1.0:
    #         # upscale to correct aspect ratio
    #         size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
    #         image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
    #
    #     return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    #
    # # reverse any processing on images so they can be written to disk or displayed to user
    # with tf.name_scope("convert_inputs"):
    #     converted_inputs = convert(inputs)
    #
    # with tf.name_scope("convert_targets"):
    #     converted_targets = convert(targets)
    #
    # with tf.name_scope("convert_outputs"):
    #     converted_outputs = convert(outputs)
    #
    # # with tf.name_scope("encode_images"):
    # #     display_fetches = {
    # #         "paths": examples.paths,
    # #         "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
    # #         "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
    # #         "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
    # #     }
    #
    # with tf.name_scope("encode_images"):
    #     display_fetches = {
    #         "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
    #         "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
    #         "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
    #     }
    #
    # # summaries
    # with tf.name_scope("inputs_summary"):
    #     tf.summary.image("inputs", converted_inputs)
    #
    # with tf.name_scope("targets_summary"):
    #     tf.summary.image("targets", converted_targets)
    #
    # with tf.name_scope("outputs_summary"):
    #     tf.summary.image("outputs", converted_outputs)
    #
    # with tf.name_scope("predict_real_summary"):
    #     tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    #
    # with tf.name_scope("predict_fake_summary"):
    #     tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))
    #
    # tf.summary.scalar("discriminator_loss", model.discrim_loss)
    # tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    # tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    #
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name + "/values", var)
    #
    # for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
    #     tf.summary.histogram(var.op.name + "/gradients", grad)
    #
    # with tf.name_scope("parameter_count"):
    #     parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    #
    # saver = tf.train.Saver(max_to_keep=1)
    #
    # logdir = output_dir if ( a.trace_freq > 0 or a.summary_freq > 0) else None
    # sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    # with sv.managed_session() as sess:
    #     print("parameter_count =", sess.run(parameter_count))
    #
    #     if a.checkpoint is not None:
    #         print("loading model from checkpoint")
    #         checkpoint = tf.train.latest_checkpoint(a.checkpoint)
    #         saver.restore(sess, checkpoint)
    #
    #     max_steps = 2**32
    #     if a.max_epochs is not None:
    #         max_steps = examples.steps_per_epoch * a.max_epochs
    #     if a.max_steps is not None:
    #         max_steps = a.max_steps
    #
    #     if a.mode == "test":
    #         # testing
    #         # at most, process the test data once
    #         start = time.time()
    #         max_steps = min(examples.steps_per_epoch, max_steps)
    #         for step in range(max_steps):
    #             results = sess.run(display_fetches)
    #             filesets = save_images(results,examples.count)
    #             for i, f in enumerate(filesets):
    #                 print("evaluated image", f["name"])
    #             # index_path = append_index(filesets)
    #         # print("wrote index at", index_path)
    #         print("rate", (time.time() - start) / max_steps)
    #     else:
    #         # training
    #         start = time.time()
    #
    #         for step in range(max_steps):
    #             def should(freq):
    #                 return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
    #
    #             options = None
    #             run_metadata = None
    #             if should(a.trace_freq):
    #                 options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #                 run_metadata = tf.RunMetadata()
    #
    #             fetches = {
    #                 "train": model.train,
    #                 "global_step": sv.global_step,
    #             }
    #
    #             if should(a.progress_freq):
    #                 fetches["discrim_loss"] = model.discrim_loss
    #                 fetches["gen_loss_GAN"] = model.gen_loss_GAN
    #                 fetches["gen_loss_L1"] = model.gen_loss_L1
    #
    #             if should(a.summary_freq):
    #                 fetches["summary"] = sv.summary_op
    #
    #             if should(a.display_freq):
    #                 fetches["display"] = display_fetches
    #
    #             results = sess.run(fetches, options=options, run_metadata=run_metadata)
    #
    #             if should(a.summary_freq):
    #                 print("recording summary")
    #                 sv.summary_writer.add_summary(results["summary"], results["global_step"])
    #
    #             if should(a.display_freq):
    #                 print("saving display images")
    #                 filesets = save_images(results["display"], examples.count, step=results["global_step"])
    #                 append_index(filesets, step=True)
    #
    #             if should(a.trace_freq):
    #                 print("recording trace")
    #                 sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])
    #
    #             if should(a.progress_freq):
    #                 # global_step will have the correct step count if we resume from a checkpoint
    #                 train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
    #                 train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
    #                 rate = (step + 1) * a.batch_size / (time.time() - start)
    #                 remaining = (max_steps - step) * a.batch_size / rate
    #                 print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
    #                 print("discrim_loss", results["discrim_loss"])
    #                 print("gen_loss_GAN", results["gen_loss_GAN"])
    #                 print("gen_loss_L1", results["gen_loss_L1"])
    #
    #             if should(a.summary_freq):
    #                 print("saving model")
    #                 saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)
    #
    #             if sv.should_stop():
    #                 break


if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
