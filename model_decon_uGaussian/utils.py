#########################
## Author: Chaochao Lu ##
#########################
import tensorflow as tf
import os
import numpy as np
from edward.models import Bernoulli
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

########################################################################################################################
########################################## basic NN components #########################################################
########################################################################################################################

def fully_connected_layer(opts, inp, out_dim, scope, reuse=tf.AUTO_REUSE):
    # input:
    #       option 1: batch_size x nsteps x width x height x channel
    #       option 2: batch_size x nsteps x dim
    #       option 3: batch_size x dim
    #       option 4: batch_size x width x height x channel
    # output:
    #       option 1 & 2: batch_size x nsteps x out_dim
    #       option 3: batch_size x dim
    #       option 4: batch_size x out_dim

    bias = opts['init_bias']
    shape = inp.get_shape().as_list()  # batch_size x nsteps x width x height x channel

    in_shape = shape[-1]

    if len(shape) == 3 or len(shape) == 5:
        inp = tf.reshape(inp, [shape[0], shape[1], -1])
        inp = tf.reshape(inp, [-1, tf.shape(inp)[2]])
        in_shape = np.prod(shape[2:])
    elif len(shape) == 4:
        inp = tf.reshape(inp, [shape[0], -1])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable('W', [in_shape, out_dim], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(bias))

    outp = tf.matmul(inp, W) + b

    if len(shape) == 3 or len(shape) == 5:
        outp = tf.reshape(outp, [shape[0], shape[1], out_dim])

    return outp

def ac_fully_connected_layer(opts, inp, out_dim, scope, reuse=tf.AUTO_REUSE, trainable=True):
    # input:
    #       option 1: batch_size x nsteps x width x height x channel
    #       option 2: batch_size x nsteps x dim
    #       option 3: batch_size x dim
    #       option 4: batch_size x width x height x channel
    # output:
    #       option 1 & 2: batch_size x nsteps x out_dim
    #       option 3: batch_size x dim
    #       option 4: batch_size x out_dim

    bias = opts['init_bias']
    shape = inp.get_shape().as_list()  # batch_size x nsteps x width x height x channel

    in_shape = shape[-1]

    if len(shape) == 3 or len(shape) == 5:
        inp = tf.reshape(inp, [shape[0], shape[1], -1])
        inp = tf.reshape(inp, [-1, tf.shape(inp)[2]])
        in_shape = np.prod(shape[2:])
    elif len(shape) == 4:
        inp = tf.reshape(inp, [shape[0], -1])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable('W', [in_shape, out_dim], tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=trainable)
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(bias), trainable=trainable)

    outp = tf.matmul(inp, W) + b

    if len(shape) == 3 or len(shape) == 5:
        outp = tf.reshape(outp, [shape[0], shape[1], out_dim])

    return outp


def conv2d_layer(opts, inp, out_dim, scope, filter_size, d_h=2, d_w=2,
                 padding='SAME', l2_norm=False, reuse= tf.AUTO_REUSE):

    # input:
    #       option 1: batch_size x nsteps x width x height x channel
    #       option 2: batch_size x width x height x channel
    # output:
    #       option 1: batch_size x nsteps x width x height x channel
    #       option 2: batch_size x width x height x channel

    bias = opts['init_bias']
    shape = inp.get_shape().as_list()

    k_h = filter_size
    k_w = filter_size

    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable('W', [k_h, k_w, shape[-1], out_dim],
                                   initializer=tf.contrib.layers.xavier_initializer_conv2d())
        if l2_norm:
            W = tf.nn.l2_normalize(W, 2)

        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(bias))

    inp = tf.reshape(inp, [-1, shape[-3], shape[-2], shape[-1]])
    conv = tf.nn.conv2d(inp, W, strides=[1, d_h, d_w, 1], padding=padding)
    outp = tf.nn.bias_add(conv, b)
    conv_shape = outp.get_shape().as_list()
    if len(shape) > 4:
        outp = tf.reshape(outp, [shape[0], shape[1], conv_shape[1], conv_shape[2], conv_shape[3]])

    return outp

def deconv2d_layer(opts, inp, out_shape, scope, filter_size, d_h=2, d_w=2, padding='SAME', reuse= tf.AUTO_REUSE):
    # input:
    #       option 1: batch_size x nsteps x width x height x channel
    #       option 2: batch_size x width x height x channel
    # output:
    #       option 1: batch_size x nsteps x width x height x channel
    #       option 2: batch_size x width x height x channel

    shape = inp.get_shape().as_list()

    k_h = filter_size
    k_w = filter_size

    out_shape = inp.get_shape().as_list()[:-3] + out_shape

    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable('W', [k_h, k_w, out_shape[-1], shape[-1]],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('b', [out_shape[-1]], initializer=tf.constant_initializer(0.0))

    inp = tf.reshape(inp, [-1, shape[-3], shape[-2], shape[-1]])
    deconv = tf.nn.conv2d_transpose(inp, W, output_shape=[tf.shape(inp)[0]] + out_shape[-3:],
                                    strides=[1, d_h, d_w, 1], padding=padding)
    deconv = tf.nn.bias_add(deconv, b)
    outp = tf.reshape(deconv, out_shape)

    return outp


########################################################################################################################
########################################## basic NN models #############################################################
########################################################################################################################

def fc_net(opts, inp, in_layers, out_layers, scope, in_activation=tf.nn.softplus, reuse=tf.AUTO_REUSE):
    h = inp
    if in_layers:
        for i, out_dim in enumerate(in_layers):
            h = fully_connected_layer(opts, h, out_dim, scope+'_fc_layer_{:d}'.format(i), reuse=reuse)
            h = in_activation(h)
            h = tf.contrib.layers.batch_norm(h, is_training=opts['model_bn_is_training'], updates_collections=None,
                                             reuse=reuse, scope=scope + '_fc_bn_{:d}'.format(i))

        if not out_layers:
            return h

    outputs = []
    for i, (out_dim, out_activation) in enumerate(out_layers):
        outp = fully_connected_layer(opts, h, out_dim, scope+'_fc_outlayer_{:d}'.format(i), reuse=reuse)
        if out_activation is not None:
            outp = out_activation(outp)
        outputs.append(outp)

    return outputs if len(outputs) > 1 else outputs[0]

def ac_fc_net(opts, inp, in_layers, out_layers, scope, is_training=True,
              in_activation=tf.nn.relu, reuse=tf.AUTO_REUSE, trainable=True):
    h = inp
    if in_layers:
        for i, out_dim in enumerate(in_layers):
            h = ac_fully_connected_layer(opts, h, out_dim, scope+'_fc_layer_{:d}'.format(i),
                                         reuse=reuse, trainable=trainable)
            h = in_activation(h)
            h = tf.layers.dropout(h, rate=opts['dropout_rate'], training=trainable & is_training)

        if not out_layers:
            return h

    outputs = []
    for i, (out_dim, out_activation) in enumerate(out_layers):
        outp = fully_connected_layer(opts, h, out_dim, scope+'_fc_outlayer_{:d}'.format(i), reuse=reuse)
        if out_activation is not None:
            outp = out_activation(outp)
        outputs.append(outp)

    return outputs if len(outputs) > 1 else outputs[0]


def encoder(opts, inp, in_channels, out_channel, scope, in_activation=tf.nn.softplus,
            out_activation=tf.nn.softplus, reuse=tf.AUTO_REUSE):
    h = inp
    filter_size = opts['filter_size']

    for i, out_dim in enumerate(in_channels):
        h = conv2d_layer(opts, h, out_dim, scope+'_conv2d_layer_{:d}'.format(i), filter_size, reuse=reuse)
        h = in_activation(h)
        h = tf.contrib.layers.batch_norm(h, is_training=opts['model_bn_is_training'], updates_collections=None,
                                         reuse=reuse, scope=scope+'_encoder_bn_{:d}'.format(i))

    for i, out_dim in enumerate(out_channel):
        h = conv2d_layer(opts, h, out_dim, scope+'_conv2d_outlayer_{:d}'.format(i), filter_size, reuse=reuse)
        if out_activation is not None:
            h = out_activation(h)

    return h


def decoder(opts, inp, in_shape, out_shape, scope, in_activation=tf.nn.softplus, reuse=tf.AUTO_REUSE):
    h = inp
    filter_size = opts['filter_size']

    for i, o_shape in enumerate(in_shape):
        h = deconv2d_layer(opts, h, o_shape, scope+'_deconv2d_layer_{:d}'.format(i), filter_size, reuse=reuse)
        h = in_activation(h)
        h = tf.contrib.layers.batch_norm(h, is_training=opts['model_bn_is_training'], updates_collections=None,
                                         reuse=reuse, scope=scope + '_decoder_bn_{:d}'.format(i))

    if not out_shape:
        return h

    outputs = []
    for i, (o_shape, out_activation) in enumerate(out_shape):
        outp = deconv2d_layer(opts, h, o_shape, scope+'_deconv2d_outlayer_{:d}'.format(i), filter_size, reuse=reuse)
        outp = out_activation(outp)
        outputs.append(outp)

    return outputs if len(outputs) > 1 else outputs[0]

########################################################################################################################
######################################### basic model functions ########################################################
########################################################################################################################

def lstm_dropout(h, dropout_prob):
    if dropout_prob > 0:
        retain_prob = 1 - dropout_prob
        h = tf.multiply(h, tf.cast(Bernoulli(probs=dropout_prob, sample_shape=h.shape), tf.float32))
        h = h / retain_prob
    return h


def gaussianKL(mu_p, cov_p, mu_q, cov_q, mask=None):
    # Both cov_p and cov_q should be positive.

    if len(mu_p.get_shape().as_list()) == 2:
        mu_p = tf.expand_dims(mu_p, 1)
        cov_p = tf.expand_dims(cov_p, 1)
        mu_q = tf.expand_dims(mu_q, 1)
        cov_q = tf.expand_dims(cov_q, 1)

    diff_mu = mu_p - mu_q
    KL = tf.log(1e-10+cov_p) - tf.log(1e-10+cov_q) - 1. + cov_q/(cov_p+1e-10) + diff_mu**2/(cov_p+1e-10)
    if mask is not None:
        KL_masked = 0.5 * tf.multiply(KL, tf.tile(mask, [1, 1, KL.get_shape().as_list()[-1]]))
    else:
        KL_masked = 0.5 * KL
    return tf.reduce_mean(tf.reduce_sum(KL_masked, axis=2))

def bernoulliKL(u_p, u_q):

    KL = u_p * (tf.log(u_p+1e-10) - tf.log(u_q+1e-10)) + (1-u_p) * (tf.log(1-u_p+1e-10) - tf.log(1-u_q+1e-10))

    return tf.reduce_mean(KL)



def gaussianNLL(data, mu, cov, mask=None):
    # data:
    #      option 1: batch_size x nsteps x dim
    #      option 2: batch_size x dim
    nll = 0.5*( tf.log(2*np.pi) + tf.log(1e-10+cov) + ((data-mu)**2/(cov+1e-10)))
    if mask is not None:
        nll = tf.multiply(nll, tf.tile(mask, [1, 1, nll.get_shape().as_list()[-1]]))

    if len(data.get_shape().as_list()) == 3:
        return tf.reduce_mean(tf.reduce_sum(nll, axis=2))
    else:
        return tf.reduce_mean(tf.reduce_sum(nll, axis=1))

def gaussianNE(mu, cov):
    # negative entropy
    ne = -0.5 * tf.log(2. * np.pi * cov + 1e-10) - 0.5
    if len(mu.get_shape().as_list()) == 3:
        return tf.reduce_mean(tf.reduce_sum(ne, axis=2))
    else:
        return tf.reduce_mean(tf.reduce_sum(ne, axis=1))

def recons_loss(cost, real, recons):
    loss = 0

    if cost == 'l2':
        # c(x,y) = ||x - y||_2
        loss = tf.reduce_sum(tf.square(real - recons), axis=[2])
        loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-10 + loss))
    elif cost == 'l2sq':
        # c(x,y) = ||x - y||_2^2
        loss = tf.reduce_sum(tf.square(real - recons), axis=[2])
        loss = 0.05 * tf.reduce_mean(loss)
    elif cost == 'l1':
        # c(x,y) = ||x - y||_1
        loss = tf.reduce_sum(tf.abs(real - recons), axis=[2])
        loss = 0.02 * tf.reduce_mean(loss)
    elif cost == 'cross_entropy':
        loss = -tf.reduce_sum(real * tf.log(1e-10 + recons) + (1 - real) * tf.log(1e-10 + 1 - recons), axis=[2])
        loss = tf.reduce_mean(loss)

    return loss

def save_mnist_plots(opts, x_gt_tr, x_gt_te, x_recons_tr, x_recons_te, train_nll, train_kl, validation_nll,
                     validation_kl, train_x_loss, validation_x_loss, train_a_loss, validation_a_loss, train_r_loss,
                     validation_r_loss, x_seq_sample, filename):
    # Generates and saves the plot of the following layout:
    #     img1 | img2 | img3 | img4 | img5
    #     img6 | img7 | img8 | img9 | img10
    #     img1    -   nll of train vs validation
    #     img2    -   kl of train vs validation
    #     img3    -   x_reconstr loss of train vs validation
    #     img4    -   a_reconstr loss of train vs validation
    #     img5    -   r_reconstr loss of train vs validation
    #     img6    -   train_x_groundtruth
    #     img7    -   train_x_reconstr
    #     img8    -   validation_x_groundtruth
    #     img9    -   validation_x_reconstr
    #     img10    -   sequence samples

    # img5: batch_size x nsteps x dim
    sample_gt_tr = []
    sample_rc_tr = []
    sample_gt_te = []
    sample_rc_te = []
    sample_gen = []

    x_gt_tr = np.reshape(x_gt_tr, [opts['batch_size'], opts['nsteps'], opts['mnist_dim'], opts['mnist_dim']])
    x_recons_tr = np.reshape(x_recons_tr, [opts['batch_size'], opts['nsteps'], opts['mnist_dim'], opts['mnist_dim']])
    x_gt_te = np.reshape(x_gt_te, [opts['batch_size'], opts['nsteps'], opts['mnist_dim'], opts['mnist_dim']])
    x_recons_te = np.reshape(x_recons_te, [opts['batch_size'], opts['nsteps'], opts['mnist_dim'], opts['mnist_dim']])
    x_seq_sample = np.reshape(x_seq_sample, [opts['batch_size'], opts['nsteps'], opts['mnist_dim'], opts['mnist_dim']])

    # maximum of sample_num is batch_size
    for i in xrange(opts['sample_num']):
        for j in xrange(opts['nsteps']):
            sample_gt_tr.append(x_gt_tr[i][j])
            sample_rc_tr.append(x_recons_tr[i][j])
            sample_gt_te.append(x_gt_te[i][j])
            sample_rc_te.append(x_recons_te[i][j])
            sample_gen.append(x_seq_sample[i][j])

    img6 = np.concatenate(sample_gt_tr, axis=1)
    img6 = np.concatenate(np.split(img6, opts['sample_num'], 1), axis=0)
    img7 = np.concatenate(sample_rc_tr, axis=1)
    img7 = np.concatenate(np.split(img7, opts['sample_num'], 1), axis=0)
    img8 = np.concatenate(sample_gt_te, axis=1)
    img8 = np.concatenate(np.split(img8, opts['sample_num'], 1), axis=0)
    img9 = np.concatenate(sample_rc_te, axis=1)
    img9 = np.concatenate(np.split(img9, opts['sample_num'], 1), axis=0)
    img10 = np.concatenate(sample_gen, axis=1)
    img10 = np.concatenate(np.split(img10, opts['sample_num'], 1), axis=0)


    dpi = 100
    height_pic = img6.shape[0]
    width_pic = img6.shape[1]
    fig_height = 4 * height_pic / float(dpi)
    fig_width = 10 * width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(2, 5)

    total_num = len(train_nll)
    x = np.arange(1, total_num + 1)

    # img1
    plt.subplot(gs[0, 0])
    plt.plot(x, train_nll, linewidth=1, color='blue', label='train_nll')
    plt.plot(x, validation_nll, linewidth=1, color='green', label='validation_nll')
    plt.legend(loc='upper right')

    # img1
    plt.subplot(gs[0, 1])
    plt.plot(x, train_kl, linewidth=1, color='blue', label='train_kl')
    plt.plot(x, validation_kl, linewidth=1, color='green', label='validation_kl')
    plt.legend(loc='upper right')

    # img3
    plt.subplot(gs[0, 2])
    plt.plot(x, train_x_loss, linewidth=1, color='blue', label='train_x_loss')
    plt.plot(x, validation_x_loss, linewidth=1, color='green', label='validation_x_loss')
    plt.legend(loc='upper right')

    # img4
    plt.subplot(gs[0, 3])
    plt.plot(x, train_a_loss, linewidth=1, color='blue', label='train_a_loss')
    plt.plot(x, validation_a_loss, linewidth=1, color='green', label='validation_a_loss')
    plt.legend(loc='upper right')

    # img5
    plt.subplot(gs[0, 4])
    plt.plot(x, train_r_loss, linewidth=1, color='blue', label='train_r_loss')
    plt.plot(x, validation_r_loss, linewidth=1, color='green', label='validation_r_loss')
    plt.legend(loc='upper right')

    # img6
    ax = plt.subplot(gs[1, 0])
    plt.imshow(img6, cmap='gray', interpolation=None, vmin=0., vmax=1.)
    plt.text(0.47, 1., 'train_x_groundtruth', ha="center", va="bottom", size=10, transform=ax.transAxes)

    # img7
    ax = plt.subplot(gs[1, 1])
    plt.imshow(img7, cmap='gray', interpolation=None, vmin=0., vmax=1.)
    plt.text(0.47, 1., 'train_x_reconstruction', ha="center", va="bottom", size=10, transform=ax.transAxes)

    # img8
    ax = plt.subplot(gs[1, 2])
    plt.imshow(img8, cmap='gray', interpolation=None, vmin=0., vmax=1.)
    plt.text(0.47, 1., 'validation_x_groundtruth', ha="center", va="bottom", size=10, transform=ax.transAxes)

    # img9
    ax = plt.subplot(gs[1, 3])
    plt.imshow(img9, cmap='gray', interpolation=None, vmin=0., vmax=1.)
    plt.text(0.47, 1., 'validation_x_reconstruction', ha="center", va="bottom", size=10, transform=ax.transAxes)

    # img10
    ax = plt.subplot(gs[1, 4])
    plt.imshow(img10, cmap='gray', interpolation=None, vmin=0., vmax=1.)
    plt.text(0.47, 1., 'counterfactual_x_generation', ha="center", va="bottom", size=10, transform=ax.transAxes)

    fig.savefig(os.path.join(opts['work_dir'], 'plots', filename), dpi=dpi)
    plt.close()











