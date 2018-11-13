from __future__ import division
import os
import time
from sys import stdout
from glob import glob
import tensorflow as tf
import numpy as np
import cv2
from six.moves import xrange

from ops import *

def add_noise(image, noise=0.1):
    with tf.name_scope("add_noise"):
        return image+tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=noise, dtype=tf.float32)

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return ((image + 1) / 2)*255

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='cityscapes_GAN',
                 checkpoint_dir=None, data=None, momentum=0.9,
                 label_smoothing=False,noise_std_dev=0.0):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        self.noise_std_dev=noise_std_dev
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1', momentum=momentum)
        self.d_bn2 = batch_norm(name='d_bn2', momentum=momentum)
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2', momentum=momentum)
        self.g_bn_e3 = batch_norm(name='g_bn_e3', momentum=momentum)
        self.g_bn_e4 = batch_norm(name='g_bn_e4', momentum=momentum)
        self.g_bn_e5 = batch_norm(name='g_bn_e5', momentum=momentum)
        self.g_bn_e6 = batch_norm(name='g_bn_e6', momentum=momentum)
        self.g_bn_e7 = batch_norm(name='g_bn_e7', momentum=momentum)
        self.g_bn_e8 = batch_norm(name='g_bn_e8', momentum=momentum)

        self.g_bn_d1 = batch_norm(name='g_bn_d1', momentum=momentum)
        self.g_bn_d2 = batch_norm(name='g_bn_d2', momentum=momentum)
        self.g_bn_d3 = batch_norm(name='g_bn_d3', momentum=momentum)
        self.g_bn_d4 = batch_norm(name='g_bn_d4', momentum=momentum)
        self.g_bn_d5 = batch_norm(name='g_bn_d5', momentum=momentum)
        self.g_bn_d6 = batch_norm(name='g_bn_d6', momentum=momentum)
        self.g_bn_d7 = batch_norm(name='g_bn_d7', momentum=momentum)

        self.data = data
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.label_smoothing = label_smoothing

        # Get the data descriptors with the shape of data coming
        data_description = data.get_data_description()
        data_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]

        # Create an iterator for the data
        self.iter_handle = tf.placeholder(tf.string, shape=[],
                                              name='training_placeholder')
        iterator = tf.data.Iterator.from_string_handle(
            self.iter_handle, *data_description)
        training_batch = iterator.get_next()

        self.build_model(training_batch['labels'], training_batch['rgb'])

    def build_model(self, input, target):
        self.real_B = preprocess(target)
        self.real_A = preprocess(input)

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        if self.label_smoothing == True:
            smoothing = tf.random_uniform(shape=[1],minval=0.9,maxval=1)
        else:
            smoothing = 1

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)*smoothing))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.real_A_sum = tf.summary.image("Input", deprocess(self.real_A)[...,::-1])
        self.real_B_sum = tf.summary.image("Target", deprocess(self.real_B)[...,::-1])
        # self.fake_B_sample_sum = tf.summary.image("fake_B_sample", deprocess(self.fake_B_sample)[...,::-1])
        self.fake_B_sum = tf.summary.image("Generated", deprocess(self.fake_B)[...,::-1])

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, args):
        """Train pix2pix"""

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                              .minimize(self.d_loss, var_list=self.d_vars)

        with tf.control_dependencies([d_optim]):
            g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                              .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # self.g_sum = tf.summary.merge([self.d__sum,self.real_A_sum,self.real_B_sum,self.fake_B_sample_sum,
        #     self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        self.sum = tf.summary.merge([self.d__sum,self.real_A_sum,self.real_B_sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)


        if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 1
        if args.warm_start == True and args.checkpoint is None:
            print("INFO: Starting warm up training")
            stdout.flush()
            measure_data = self.data.get_measureset()
            measure_iterator = measure_data.repeat(1000).batch(args.batch_size).make_one_shot_iterator()
            measure_handle = self.sess.run(measure_iterator.string_handle())
            start_time = time.time()
            while True:
                if np.mod(counter, args.num_print) == 1:
                    try:
                        _, summary_str, d_l, g_l = self.sess.run([g_optim, self.sum, self.d_loss, self.g_loss],
                                                       feed_dict={ self.iter_handle: measure_handle })
                    except tf.errors.OutOfRangeError:
                        print("INFO: Done with warm up training")
                        break

                    self.writer.add_summary(summary_str, counter)
                    print("Step: [%2d] rate: %4.4f steps/sec, d_loss: %.8f, g_loss: %.8f" \
                        % (counter,args.batch_size*counter/(time.time() - start_time), d_l, g_l))
                    stdout.flush()
                else:
                    try:
                        self.sess.run(g_optim,feed_dict={ self.iter_handle: measure_handle })
                    except tf.errors.OutOfRangeError:
                        print("INFO: Done with warm up training")
                        break
                counter += 1

        # for epoch in xrange(args.epoch):
        for epoch in range(0,1):
            input_data = self.data.get_trainset()
            data_iterator = input_data.repeat(args.max_epochs).batch(args.batch_size).make_one_shot_iterator()
            data_handle = self.sess.run(data_iterator.string_handle())
            counterTrain = 1
            start_time = time.time()
            # for idx in xrange(0, batch_idxs):
            while True:
                if np.mod(counterTrain, args.num_print) == 1:
                    try:
                        _, summary_str, d_l, g_l = self.sess.run([g_optim, self.sum, self.d_loss, self.g_loss],
                                                       feed_dict={ self.iter_handle: data_handle })
                    except tf.errors.OutOfRangeError:
                        print("INFO: Done with all steps")
                        self.save(self.checkpoint_dir, counterTrain)
                        break

                    self.writer.add_summary(summary_str, counterTrain+counter-1)
                    print("Step: [%2d] rate: %4.4f steps/sec, d_loss: %.8f, g_loss: %.8f" \
                        % (counterTrain,args.batch_size*counterTrain/(time.time() - start_time), d_l, g_l))
                    stdout.flush()
                else:
                    try:
                        self.sess.run(g_optim,feed_dict={ self.iter_handle: data_handle })
                    except tf.errors.OutOfRangeError:
                        print("INFO: Done with all training steps")
                        self.save(self.checkpoint_dir, counterTrain)
                        break
                counterTrain += 1

        pred_array = np.zeros((15,2))

        if not os.path.exists(os.path.join(args.file_output_dir,str(args.RUN_id))):
            os.makedirs(os.path.join(args.file_output_dir,str(args.RUN_id)))

        validation_data = self.data.get_validation_set()
        valid_iterator = validation_data.batch(args.batch_size).make_one_shot_iterator()
        valid_handle = self.sess.run(valid_iterator.string_handle())
        counter = 1
        while True:
            # Update D network
            try:
                outImage, real_val, fake_val = self.sess.run([self.fake_B,self.D,self.D_],
                                               feed_dict={ self.iter_handle: valid_handle })
            except tf.errors.OutOfRangeError:
                print("INFO: Done with all validation steps")
                break
            # print(real_val[0],fake_val[0])
            filename = str(args.RUN_id)+"_realfield" + str(counter) + ".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.RUN_id),filename), cv2.resize(255*real_val[0,:,:,0],(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))
            filename = str(args.RUN_id)+"_fakefield" + str(counter) + ".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.RUN_id),filename), cv2.resize(255*fake_val[0,:,:,0],(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))
            pred_array[counter-1,:] = [np.mean(real_val[0]),np.mean(fake_val[0])]
            filename = str(args.RUN_id)+"_validation" + str(counter) + ".png"
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.RUN_id),filename), deprocess(outImage[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            counter += 1
        print(pred_array)
        return pred_array


    def validate(self, args):
        """Train pix2pix"""
        pred_array = np.zeros((15,2))
        counter = 1

        self.load(os.path.join(args.EXP_OUT,str(args.checkpoint)))

        if not os.path.exists(os.path.join(args.file_output_dir,str(args.checkpoint))):
            os.makedirs(os.path.join(args.file_output_dir,str(args.checkpoint)))

        # for epoch in xrange(args.epoch):
        for epoch in range(0,1):
            validation_data = self.data.get_validation_set()
            valid_iterator = validation_data.batch(args.batch_size).make_one_shot_iterator()
            valid_handle = self.sess.run(valid_iterator.string_handle())
            # for idx in xrange(0, batch_idxs):
            while True:
                # Update D network
                try:
                    outImage, inpt, target, real_val, fake_val = self.sess.run([self.fake_B,self.real_A,self.real_B,self.D,self.D_],
                                                   feed_dict={ self.iter_handle: valid_handle })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all steps")
                    break
                pred_array[counter-1,:] = [np.mean(real_val[0]),np.mean(fake_val[0])]

                # Save the 30 x 30 output of the discriminator
                filename = str(args.checkpoint)+"_realfield" + str(counter) + ".png"
                cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), cv2.resize(255*real_val[0,:,:,0],(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))
                filename = str(args.checkpoint)+"_fakefield" + str(counter) + ".png"
                cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), cv2.resize(255*fake_val[0,:,:,0],(args.input_image_size,args.input_image_size),interpolation=cv2.INTER_NEAREST))
                # Save the output of the generator
                filename = str(args.checkpoint)+"_validation" + str(counter) + ".png"
                cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), deprocess(outImage[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                # If needed, save the input and target
                if args.val_target_output == True:
                    filename = "input_validation" + str(counter) + ".png"
                    cv2.imwrite(os.path.join(args.file_output_dir,filename), deprocess(inpt[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    filename = "target_validation" + str(counter) + ".png"
                    cv2.imwrite(os.path.join(args.file_output_dir,filename), deprocess(target[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                counter += 1
            print(pred_array)
            return pred_array

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv',pad="VALID")))
            # h3 is (31 x 31 x self.df_dim*8)
            h4 = conv2d(h3, 1, d_h=1, d_w=1, name='d_h4_conv',pad="VALID")
            # h4 is (30 x 30 x 1)
            # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            # print(h4.shape)

            return tf.nn.sigmoid(h4), h4
    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(add_noise(image), self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(add_noise(e1,self.noise_std_dev)), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(add_noise(e2,self.noise_std_dev)), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(add_noise(e3,self.noise_std_dev)), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(add_noise(e4,self.noise_std_dev)), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(add_noise(e5,self.noise_std_dev)), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(add_noise(e6,self.noise_std_dev)), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(add_noise(e7,self.noise_std_dev)), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(add_noise(e8,self.noise_std_dev)),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(add_noise(d1,self.noise_std_dev)),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(add_noise(d2,self.noise_std_dev)),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(add_noise(d3,self.noise_std_dev)),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(add_noise(d4,self.noise_std_dev)),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(add_noise(d5,self.noise_std_dev)),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(add_noise(d6,self.noise_std_dev)),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(add_noise(d7,self.noise_std_dev)),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)
    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)


    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, checkpoint)
        return True
