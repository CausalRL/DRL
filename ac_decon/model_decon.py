#########################
## Author: Chaochao Lu ##
#########################
import numpy as np
import tensorflow as tf
import os
import time

from utils import *

class Model_Decon(object):

    def __init__(self, sess, opts):
        self.config = None
        self.sess = sess
        self.saver = None

        self.opts = opts
        np.random.seed(self.opts['seed'])
        self.fb = 'forward'
        self.u = None

        self.u_x_list = []
        self.u_a_list = []
        self.u_r_list = []

########################################################################################################################
############################################# create generative model ##################################################
########################################################################################################################

    def p_z(self):
        mu_z = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        cov_z = tf.ones([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        eps = tf.random_normal((self.opts['batch_size'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z = mu_z + tf.multiply(eps, tf.sqrt(1e-8 + cov_z))

        return z

    def p_x_g_z_u(self, z, u):
        # z:
        #   option 1: batch_size x nsteps x dim
        #   option 2: batch_size x dim
        # u: batch_size x u_dim

        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['pxgz_net_layers'], self.opts['pxgz_net_outlayers'], 'pxgz_net')
        u_fea = fc_net(self.opts, u, self.opts['pxgu_net_layers'], self.opts['pxgu_net_outlayers'], 'pxgu_net')

        if len(z.get_shape().as_list()) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        if self.opts['is_conv']:
            # z_fea: batch_size x [nsteps] x (4 x 4 x 32)
            z_fea = fc_net(self.opts, zu_fea, self.opts['pxgzu_prenet_layers'],
                           self.opts['pxgzu_prenet_outlayers'], 'pxgz_prenet')
            z_fea = tf.reshape(z_fea, z_fea.get_shape().as_list()[:-1] + [4, 4, 32])

            # mu, sigma: batch_size x [nsteps] x 28 x 28 x 1
            mu, sigma = decoder(self.opts, z_fea, self.opts['pxgzu_in_shape'],
                                self.opts['pxgzu_out_shape'], 'pxgzu_conv_net')

            # mu, sigma: batch_size x [nsteps] x x_dim
            mu = tf.reshape(mu, mu.get_shape().as_list()[:-3] + [-1])
            sigma = tf.reshape(sigma, sigma.get_shape().as_list()[:-3] + [-1])

        else:
            mu, sigma = fc_net(self.opts, zu_fea, self.opts['pxgzu_net_layers'],
                               self.opts['pxgzu_net_outlayers'], 'pxgzu_net')

        return mu, sigma

    def p_a_g_z_u(self, z, u):
        # z:
        #   option 1: batch_size x nsteps x dim
        #   option 2: batch_size x dim
        # u: batch_size x u_dim

        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['pagz_net_layers'], self.opts['pagz_net_outlayers'], 'pagz_net')
        u_fea = fc_net(self.opts, u, self.opts['pagu_net_layers'], self.opts['pagu_net_outlayers'], 'pagu_net')

        if len(z.get_shape().as_list()) > 2:
            zu_fea = tf.concat([z_fea, u_fea], 2)
        else:
            zu_fea = tf.concat([z_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zu_fea, self.opts['pagzu_net_layers'],
                           self.opts['pagzu_net_outlayers'], 'pagzu_net')
        mu = mu * self.opts['a_range']

        return mu, sigma

    def p_r_g_z_a_u(self, z, a, u):

        if len(z.get_shape().as_list()) > 2:
            u = tf.expand_dims(u, 1)
            u = tf.tile(u, [1, z.get_shape().as_list()[1], 1])

        z_fea = fc_net(self.opts, z, self.opts['prgz_net_layers'], self.opts['prgz_net_outlayers'], 'prgz_net')
        a_fea = fc_net(self.opts, a, self.opts['prga_net_layers'], self.opts['prga_net_outlayers'], 'prga_net')
        u_fea = fc_net(self.opts, u, self.opts['prgu_net_layers'], self.opts['prgu_net_outlayers'], 'pagu_net')

        if len(z.get_shape().as_list()) > 2:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 2)
        else:
            zau_fea = tf.concat([z_fea, a_fea, u_fea], 1)

        mu, sigma = fc_net(self.opts, zau_fea, self.opts['prgzau_net_layers'],
                           self.opts['prgzau_net_outlayers'], 'prgzau_net')

        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']
        return mu, sigma

    def p_z_g_z_a(self, z, a):

        z_fea = fc_net(self.opts, z, self.opts['pzgz_net_layers'], self.opts['pzgz_net_outlayers'], 'pzgz_net')
        a_fea = fc_net(self.opts, a, self.opts['pzga_net_layers'], self.opts['pzga_net_outlayers'], 'pzga_net')

        if len(z.get_shape().as_list()) > 2:
            az_fea = tf.concat([z_fea, a_fea], 2)
        else:
            az_fea = tf.concat([z_fea, a_fea], 1)

        h_az_fea = fc_net(self.opts, az_fea, self.opts['pzgza_net_layers'],
                          self.opts['pzgza_net_outlayers'], 'pzgza_net')
        h_mu = fc_net(self.opts, h_az_fea, self.opts['pzgza_mu_net_layers'],
                      self.opts['pzgza_mu_net_outlayers'], 'pzgza_mu_net')

        if self.opts['gated']:
            hg_az_fea = fc_net(self.opts, az_fea, self.opts['pzgza_pregate_net_layers'],
                               self.opts['pzgza_pregate_net_outlayers'], 'pzgza_pregate_net')
            gate = fc_net(self.opts, hg_az_fea, self.opts['pzgza_gate_net_layers'],
                          self.opts['pzgza_gate_net_outlayers'], 'pzgza_gate_net')
            mu = gate * h_mu + (1 - gate) * fc_net(self.opts, az_fea, self.opts['pzgza_gate_mu_net_layers'],
                                                   self.opts['pzgza_gate_mu_net_outlayers'], 'pzgza_gate_mu_net')
        else:
            mu = h_mu

        sigma = fc_net(self.opts, h_az_fea, self.opts['pzgza_sigma_net_layers'],
                       self.opts['pzgza_sigma_net_outlayers'], 'pzgza_sigma_net')

        return mu, sigma

########################################################################################################################
############################################# create inference/recognition model #######################################
########################################################################################################################
    def lstm_cell(self, prev, current):
        # h_prev, c_prev: batch_size x lstm_dim
        h_prev = prev[0]
        c_prev = prev[1]
        # x_current: batch_size x lstm_dim
        x_current = current[0]
        # mask: batch_size x lstm_dim
        mask = current[1]

        with tf.variable_scope('lstm_cell_'+self.fb, reuse=tf.AUTO_REUSE):
            w_i = tf.get_variable('w_i', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_i = tf.get_variable('b_i', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))
            w_f = tf.get_variable('w_f', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_f = tf.get_variable('b_f', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))
            w_o = tf.get_variable('w_o', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable('b_o', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))
            w_c = tf.get_variable('w_c', [2*self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable('b_c', [self.opts['lstm_dim']], tf.float32,
                                  initializer=tf.constant_initializer(self.opts['init_bias']))

        # concat_x_h: batch_size x (lstm_dim + lstm_dim)
        concat_x_h = tf.concat([x_current, h_prev], 1)
        # i, f, o, c: batch_size x lstm_dim
        i = tf.sigmoid(tf.matmul(concat_x_h, w_i) + b_i)
        f = tf.sigmoid(tf.matmul(concat_x_h, w_f) + b_f)
        o = tf.sigmoid(tf.matmul(concat_x_h, w_o) + b_o)
        c = tf.tanh(tf.matmul(concat_x_h, w_c) + b_c)

        c_new = tf.multiply(f, c_prev) + tf.multiply(i, c)
        c = tf.multiply(c_new, mask) + tf.multiply(1 - mask, c_prev)
        h_new = tf.multiply(o, tf.tanh(c))
        h = tf.multiply(h_new, mask) + tf.multiply(1 - mask, h_prev)

        return h, c

    def lstm_net(self, lstm_input, suffix, mask=None):
        # lstm_input: batch_size x nsteps x 100
        lstm_input = tf.transpose(lstm_input, [1, 0, 2])
        # lstm_input: nsteps x batch_size x 100
        lstm_input_shape = lstm_input.get_shape().as_list()

        # lstm_embed: nsteps x batch_size x lstm_dim
        lstm_embed = fc_net(self.opts, lstm_input, self.opts['lstm_net_layers'],
                            self.opts['lstm_net_outlayers'], 'lstm_net')

        if mask is None:
            mask = tf.ones([lstm_input_shape[0], self.opts['batch_size'], self.opts['lstm_dim']])
        else:
            mask = tf.tile(mask, [1, 1, self.opts['lstm_dim']])
            mask = tf.transpose(mask, [1, 0, 2])

        if suffix == 'R' or suffix == 'UR':
            self.fb = 'backward' + '_' + suffix

            lstm_embed = tf.reverse(lstm_embed, [0])
            mask = tf.reverse(mask, [0])
            # lstm_embed: nsteps x 1 x batch_size x lstm_dim
            lstm_embed = lstm_embed[:, tf.newaxis]
            # mask: nsteps x 1 x batch_size x lstm_dim
            mask = mask[:, tf.newaxis]
            # lm_concat: nsteps x 2 x batch_size x lstm_dim
            lm_concat = tf.concat([lstm_embed, mask], 1)
            # lm_split: nsteps x 1 x 2 x batch_size x lstm_dim
            lm_split = tf.split(lm_concat, lstm_input_shape[0], 0)
            lm_list = []
            for i in xrange(lstm_input_shape[0]):
                lm_list.append(tf.reshape(lm_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

            elements = tf.convert_to_tensor(lm_list)

            output_backward = tf.scan(
                self.lstm_cell,
                elements,
                initializer=(tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]),
                             tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]))
            )

            lstm_output = output_backward[0]
            lstm_output = tf.reverse(lstm_output, [0])
        else:
            self.fb = 'forward' + '_' + suffix

            # lstm_embed: nsteps x 1 x batch_size x lstm_dim
            lstm_embed = lstm_embed[:, tf.newaxis]
            # mask: nsteps x 1 x batch_size x lstm_dim
            mask = mask[:, tf.newaxis]
            # lm_concat: nsteps x 2 x batch_size x lstm_dim
            lm_concat = tf.concat([lstm_embed, mask], 1)
            # lm_split: nsteps x 1 x 2 x batch_size x lstm_dim
            lm_split = tf.split(lm_concat, lstm_input_shape[0], 0)
            lm_list = []
            for i in xrange(lstm_input_shape[0]):
                lm_list.append(tf.reshape(lm_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

            elements = tf.convert_to_tensor(lm_list)

            # elements[0] (lstm_embed): batch_size x lstm_dim
            # elements[1] (mask): batch_size x lstm_dim
            output_forward = tf.scan(
                self.lstm_cell,
                elements,
                initializer=(tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]),
                             tf.zeros([self.opts['batch_size'], self.opts['lstm_dim']]))
            )

            # lstm_output: nsteps x batch_size x lstm_dim
            lstm_output = output_forward[0]

        return lstm_dropout(lstm_output, self.opts['lstm_dropout_prob'])

    def st_approx(self, prev, current):
        # z_prev: batch_size x z_dim
        z_prev = prev[0]
        # h_current: batch_size x lstm_dim
        h_current = current[0]
        # a_prev: batch_size x z_dim
        a_prev = current[1]

        with tf.variable_scope('lstm_st_approx', reuse=tf.AUTO_REUSE):
            w_st_z = tf.get_variable('w_st_z', [self.opts['z_dim'], self.opts['lstm_dim']], tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_st_z = tf.get_variable('b_st_z', [self.opts['lstm_dim']], tf.float32,
                                     initializer=tf.constant_initializer(self.opts['init_bias']))
            w_st_a = tf.get_variable('w_st_a', [self.opts['lstm_dim'], self.opts['lstm_dim']], tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_st_a = tf.get_variable('b_st_a', [self.opts['lstm_dim']], tf.float32,
                                     initializer=tf.constant_initializer(self.opts['init_bias']))
            w_st_mu = tf.get_variable('w_st_mu', [self.opts['lstm_dim'], self.opts['z_dim']], tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_st_mu = tf.get_variable('b_st_mu', [self.opts['z_dim']], tf.float32,
                                      initializer=tf.constant_initializer(self.opts['init_bias']))
            w_st_cov = tf.get_variable('w_st_cov', [self.opts['lstm_dim'], self.opts['z_dim']], tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_st_cov = tf.get_variable('b_st_cov', [self.opts['z_dim']], tf.float32,
                                       initializer=tf.constant_initializer(self.opts['init_bias']))
        # h_next: batch_size x lstm_dim
        h_next = tf.tanh(tf.matmul(z_prev, w_st_z) + b_st_z + tf.matmul(a_prev, w_st_a) + b_st_a)
        # h_combined: batch_size x lstm_dim
        h_combined = 0.5 * (h_current + h_next)
        # mu, cov, z: batch_size x z_dim
        mu = tf.matmul(h_combined, w_st_mu) + b_st_mu
        cov = tf.nn.softplus(tf.matmul(h_combined, w_st_cov) + b_st_cov)

        eps = tf.random_normal(mu.get_shape(), 0., 1., dtype=tf.float32)
        z = mu + tf.multiply(eps, tf.sqrt(1e-8 + cov))

        return z, mu, cov


    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        # x_seq should be: batch_size x nsteps x dim
        x_seq_dim = 3
        # x_seq: batch_size x dim
        if len(x_seq.get_shape().as_list()) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)

            x_seq_dim = 2

        # x_seq: batch_size x nsteps x dim
        if self.opts['is_conv']:
            x_reshape = tf.reshape(x_seq, x_seq.get_shape().as_list()[:-1] + [28, 28, 1])
            # x_reshape: batch_size x nsteps x 28 x 28 x 1
            x_encoded = encoder(self.opts, x_reshape, self.opts['qzgx_in_channels'],
                                self.opts['qzgx_out_channel'], 'qzgx_conv_net')
            x_fea = fc_net(self.opts, x_encoded, self.opts['qzgx_encoded_net_layers'],
                           self.opts['qzgx_encoded_net_outlayers'], 'qzgx_encoded_net')
        else:
            x_fea = fc_net(self.opts, x_seq, self.opts['qzgx_net_layers'],
                           self.opts['qzgx_net_outlayers'], 'qzgx_net')

        # a_fea: batch_size x nsteps x 100
        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'], self.opts['qzga_net_outlayers'], 'qzga_net')
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'], self.opts['qzgr_net_outlayers'], 'qzgr_net')

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        # xar_fea: batch_size x nsteps x 100
        xar_fea = fc_net(self.opts, concat_xar, self.opts['qzgxar_net_layers'],
                         self.opts['qzgxar_net_outlayers'], 'qzgxar_net')

        # xar_fea: batch_size x nsteps x 100
        h_r = self.lstm_net(xar_fea, 'R', mask)

        if self.opts['inference_model_type'] == 'LR':
            h_l = self.lstm_net(xar_fea, 'L', mask)
            # h: nsteps x batch_size x lstm_dim
            h = (h_r + h_l)/2.
        else:
            # h: nsteps x batch_size x lstm_dim
            h = h_r

        z_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']], tf.float32)
        mu_0 = tf.zeros([self.opts['batch_size'], self.opts['z_dim']])
        cov_0 = tf.ones([self.opts['batch_size'], self.opts['z_dim']])

        # h: nsteps x 1 x batch_size x lstm_dim
        h = h[:, tf.newaxis]
        # a_fea: batch_size x nsteps x lstm_dim
        a_fea = fc_net(self.opts, a_fea, self.opts['qagh_net_layers'], self.opts['qagh_net_outlayers'], 'qagh_net')
        a_fea = tf.transpose(a_fea, [1, 0, 2])
        # a_fea: nsteps x batch_size x lstm_dim
        a_fea = tf.concat([tf.ones([1, self.opts['batch_size'], tf.shape(a_fea)[2]]), a_fea[:-1, :, :]], 0)
        # a_fea: nsteps x 1 x batch_size x lstm_dim
        a_fea = a_fea[:, tf.newaxis]

        # ha_concat: nsteps x 2 x batch_size x lstm_dim
        ha_concat = tf.concat([h, a_fea], 1)
        # ha_split: nsteps x 1 x 2 x batch_size x lstm_dim
        ha_split = tf.split(ha_concat, x_seq.get_shape().as_list()[1], 0)
        ha_list = []
        for i in xrange(x_seq.get_shape().as_list()[1]):
            ha_list.append(tf.reshape(ha_split[i], [2, self.opts['batch_size'], self.opts['lstm_dim']]))

        elements = tf.convert_to_tensor(ha_list)

        output_q = tf.scan(
            self.st_approx,
            elements,
            initializer=(z_0, mu_0, cov_0)
        )

        z = tf.transpose(output_q[0], [1, 0, 2])
        mu = tf.transpose(output_q[1], [1, 0, 2])
        cov = tf.transpose(output_q[2], [1, 0, 2])

        # x_seq: batch_size x nsteps x z_dim
        if x_seq_dim == 2:
            z = tf.squeeze(z, [1])
            mu = tf.squeeze(mu, [1])
            cov = tf.squeeze(cov, [1])

        return z, mu, cov

    def q_u_g_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        # x_seq should be: batch_size x nsteps x dim
        x_seq_dim = 3
        # x_seq: batch_size x dim
        if len(x_seq.get_shape().as_list()) == 2:
            x_seq = tf.expand_dims(x_seq, 1)
            a_seq = tf.expand_dims(a_seq, 1)
            r_seq = tf.expand_dims(r_seq, 1)

            x_seq_dim = 2

        # x_seq: batch_size x nsteps x dim
        if self.opts['is_conv']:
            x_reshape = tf.reshape(x_seq, x_seq.get_shape().as_list()[:-1] + [28, 28, 1])

            x_encoded = encoder(self.opts, x_reshape, self.opts['qzgx_in_channels'],
                                self.opts['qzgx_out_channel'], 'qzgx_conv_net', reuse=True)
            x_fea = fc_net(self.opts, x_encoded, self.opts['qzgx_encoded_net_layers'],
                           self.opts['qzgx_encoded_net_outlayers'], 'qzgx_encoded_net', reuse=True)
        else:
            x_fea = fc_net(self.opts, x_seq, self.opts['qugx_net_layers'],
                           self.opts['qugx_net_outlayers'], 'qugx_net')

        # a_fea: batch_size x nsteps x 100
        a_fea = fc_net(self.opts, a_seq, self.opts['qzga_net_layers'],
                       self.opts['qzga_net_outlayers'], 'qzga_net', reuse=True)
        r_fea = fc_net(self.opts, r_seq, self.opts['qzgr_net_layers'],
                       self.opts['qzgr_net_outlayers'], 'qzgr_net', reuse=True)

        concat_xar = tf.concat([x_fea, a_fea, r_fea], 2)
        # xar_fea: batch_size x nsteps x 100
        xar_fea = fc_net(self.opts, concat_xar, self.opts['qzgxar_net_layers'],
                         self.opts['qzgxar_net_outlayers'], 'qzgxar_net', reuse=True)

        # xar_fea: batch_size x nsteps x 100
        h_r = self.lstm_net(xar_fea, 'UR', mask)
        h_r = tf.reverse(h_r, [0])

        if self.opts['inference_model_type'] == 'LR':
            h_l = self.lstm_net(xar_fea, 'UL', mask)
            # h: nsteps x batch_size x lstm_dim
            h_l = tf.reverse(h_l, [0])

            h = (h_r[0] + h_l[0])/2.
        else:
            # h: nsteps x batch_size x lstm_dim
            h = h_r[0]


        # h_trans: batch_size x nsteps x lstm_dim
        h_trans = tf.reshape(h, [self.opts['batch_size'], -1])


        # Case 1: u is a Gaussian
        # mu, cov = fc_net(self.opts, h_trans, self.opts['qugh_net_layers'],
        #                  self.opts['qugh_net_outlayers'], 'qugh_net')
        # return mu, cov

        # Case 2: u is a Bernoulli
        logits = fc_net(self.opts, h_trans, self.opts['qugh_net_layers'],
                        self.opts['qugh_net_outlayers'], 'qugh_net')
        prediction = tf.nn.sigmoid(logits)

        return logits, prediction

    def q_a_g_x(self, x):
        # x: batch_size x nsteps x x_dim
        if self.opts['is_conv']:
            x_reshape = tf.reshape(x, x.get_shape().as_list()[:-1] + [28, 28, 1])
            # x_reshape: batch_size x nsteps x 28 x 28 x 1
            x_encoded = encoder(self.opts, x_reshape, self.opts['qagx_in_channels'], self.opts['qagx_out_channel'],
                                'qagx_conv_net')
            mu, sigma = fc_net(self.opts, x_encoded, self.opts['qagx_encoded_net_layers'],
                               self.opts['qagx_encoded_net_outlayers'], 'qagx_encoded_net')
        else:
            mu, sigma = fc_net(self.opts, x, self.opts['qagx_net_layers'],
                               self.opts['qagx_net_outlayers'], 'qagx_net')
        mu = mu * self.opts['a_range']
        return mu, sigma

    def q_r_g_x_a(self, x, a):
        # x: batch_size x nsteps x x_dim
        if self.opts['is_conv']:
            x_reshape = tf.reshape(x, x.get_shape().as_list()[:-1] + [28, 28, 1])
            # x_reshape: batch_size x nsteps x 28 x 28 x 1
            x_encoded = encoder(self.opts, x_reshape, self.opts['qrgx_in_channels'], self.opts['qrgx_out_channel'],
                                'qrgx_conv_net')
            x_fea = fc_net(self.opts, x_encoded, self.opts['qrgx_encoded_net_layers'],
                           self.opts['qrgx_encoded_net_outlayers'], 'qrgx_encoded_net')
        else:
            x_fea = fc_net(self.opts, x, self.opts['qrgx_net_layers'], self.opts['qrgx_net_outlayers'], 'qrgx_net')

        a_fea = fc_net(self.opts, a, self.opts['qrga_net_layers'], self.opts['qrga_net_outlayers'], 'qrga_net')

        if len(x.get_shape().as_list()) > 2:
            ax_fea = tf.concat([x_fea, a_fea], 2)
        else:
            ax_fea = tf.concat([x_fea, a_fea], 1)

        mu, sigma = fc_net(self.opts, ax_fea, self.opts['qrgxa_net_layers'],
                           self.opts['qrgxa_net_outlayers'], 'qrgxa_net')
        mu = mu * (self.opts['r_range_upper'] - self.opts['r_range_lower']) + self.opts['r_range_lower']

        return mu, sigma


########################################################################################################################
############################################# create neg_elbo ##########################################################
########################################################################################################################
    def neg_elbo(self, x_seq, a_seq, r_seq, u_seq, anneal=1, mask=None):
        # z_q, mu_q, cov_q: batch_size x nsteps x z_dim
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)  # nsteps of z_q, mu_q, cov_q: 1 to T

        eps = tf.random_normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))

        mu_p, cov_p = self.p_z_g_z_a(z_q_samples, a_seq)                          # nsteps of mu_p, cov_p: 2 to T+1

        mu_prior = tf.concat([tf.zeros([self.opts['batch_size'], 1, self.opts['z_dim']]), mu_p[:, :-1, :]], 1)
        cov_prior = tf.concat([tf.ones([self.opts['batch_size'], 1, self.opts['z_dim']]), cov_p[:, :-1, :]], 1)

        kl_divergence = gaussianKL(mu_prior, cov_prior, mu_q, cov_q, mask)

        # Case 1: KL divergence of u when u is a Gaussian
        # mu_u, cov_u = self.q_u_g_x_a_r(x_seq, a_seq, r_seq, mask)
        #
        # mu_u_prior = tf.zeros([self.opts['batch_size'], self.opts['u_dim']])
        # cov_u_prior = tf.ones([self.opts['batch_size'], self.opts['u_dim']])
        #
        # u_kl_divergence = gaussianKL(mu_u_prior, cov_u_prior, mu_u, cov_u, mask)
        #
        # mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, mu_u)
        # mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, mu_u)
        # mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, mu_u)

        # Case 2: KL divergence of u when u is a Bernoulli
        u_logits, u_prediction = self.q_u_g_x_a_r(x_seq, a_seq, r_seq, mask)
        u_prior = 0.5 * tf.ones([self.opts['batch_size'], self.opts['u_dim']])
        u_kl_divergence = bernoulliKL(u_prior, u_prediction)


        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, u_prediction)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, u_prediction)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, u_prediction)

        mu_qagx, cov_qagx = self.q_a_g_x(x_seq)
        mu_qrgxa, cov_qrgxa = self.q_r_g_x_a(x_seq, a_seq)

        nll_pxgz = gaussianNLL(x_seq, mu_pxgz, cov_pxgz, mask)
        nll_pagz = gaussianNLL(a_seq, mu_pagz, cov_pagz, mask)
        nll_prgza = gaussianNLL(r_seq, mu_prgza, cov_prgza, mask)

        nll_qagx = gaussianNLL(a_seq, mu_qagx, cov_qagx, mask)
        nll_qrgxa = gaussianNLL(r_seq, mu_qrgxa, cov_qrgxa, mask)

        #  Both nll_pagz and nll_qagx are not necessary when a does not depend on z/x in the data
        nll = nll_pxgz + nll_pagz + nll_prgza + anneal * kl_divergence + nll_qagx + nll_qrgxa + u_kl_divergence

        u_prediction = tf.expand_dims(u_prediction, 1)
        u_prediction = tf.tile(u_prediction, [1, u_seq.get_shape().as_list()[1], 1])

        correct_prediction = tf.equal(tf.round(u_prediction), u_seq)
        u_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        return nll, kl_divergence, u_kl_divergence, u_accuracy

########################################################################################################################
############################################# inference on the model ###################################################
########################################################################################################################
    def update_u(self, x_prev, a_prev, r_prev):
        self.u_x_list.append(x_prev)
        self.u_a_list.append(a_prev)
        self.u_r_list.append(r_prev)

        x_seq = tf.transpose(tf.convert_to_tensor(self.u_x_list), [1, 0, 2])
        a_seq = tf.transpose(tf.convert_to_tensor(self.u_a_list), [1, 0, 2])
        r_seq = tf.transpose(tf.convert_to_tensor(self.u_r_list), [1, 0, 2])

        self.u, _ = self.q_u_g_x_a_r(x_seq, a_seq, r_seq)

    def clear_u(self):
        self.u_x_list = []
        self.u_a_list = []
        self.u_r_list = []

    def gen_st_approx(self, prev, current):
        # z_prev: batch_size x dim
        z_prev = prev[0]

        x_prev, _ = self.p_x_g_z_u(z_prev, self.u)

        # Case 1: a is given by a policy
        # a_prev, _ = self.p_a_g_z_u(z_prev, self.u)

        # Case 2: a is uniformly selected
        a_prev = 2.*(2.*tf.random_uniform((self.opts['batch_size'], self.opts['a_dim']), 0., 1., dtype=tf.float32)-1)

        r_prev, _ = self.p_r_g_z_a_u(z_prev, a_prev, self.u)

        z_current_mu, _ = self.p_z_g_z_a(z_prev, a_prev)

        self.update_u(x_prev, a_prev, r_prev)

        return z_current_mu, x_prev, a_prev, r_prev


    def gen_xar_seq_g_z(self, z_0):
        z_0_shape = z_0.get_shape().as_list()
        if len(z_0_shape) > 2:
            z_0 = tf.reshape(z_0, [z_0_shape[0], z_0_shape[2]])

        # z_0: batch_size x z_dim
        output_xar = tf.scan(
            self.gen_st_approx,
            tf.range(self.opts['nsteps']),
            initializer=(z_0, tf.zeros([self.opts['batch_size'], self.opts['x_dim']]),
                         tf.zeros([self.opts['batch_size'], self.opts['a_dim']]),
                         tf.zeros([self.opts['batch_size'], self.opts['r_dim']]))
        )

        self.clear_u()

        return tf.transpose(output_xar[1], [1, 0, 2])


    def recons_xar_seq_g_xar_seq(self, x_seq, a_seq, r_seq, mask):
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)  # nsteps of z_q, mu_q, cov_q: 1 to T

        eps = tf.random_normal((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']),
                               0., 1., dtype=tf.float32)
        z_q_samples = mu_q + tf.multiply(eps, tf.sqrt(1e-8 + cov_q))

        mu_u, cov_u = self.q_u_g_x_a_r(x_seq, a_seq, r_seq, mask)

        mu_pxgz, cov_pxgz = self.p_x_g_z_u(z_q_samples, mu_u)
        mu_pagz, cov_pagz = self.p_a_g_z_u(z_q_samples, mu_u)
        mu_prgza, cov_prgza = self.p_r_g_z_a_u(z_q_samples, a_seq, mu_u)

        return mu_pxgz, mu_pagz, mu_prgza

    def gen_z_g_x(self, x):
        # x: batch_size x nsteps x_dim

        a, _ = self.q_a_g_x(x)
        r, _ = self.q_r_g_x_a(x, a)
        self.u, _ = self.q_u_g_x_a_r(x, a, r)

        _, z, _ = self.q_z_g_z_x_a_r(x, a, r)

        return z


########################################################################################################################
############################################# train the model ##########################################################
########################################################################################################################
    def train_model(self, data):
        batch_num = np.floor(data.train_num / self.opts['batch_size']).astype(int)
        counter = self.opts['counter_start']

        x_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['x_dim']])
        a_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['a_dim']])
        r_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['r_dim']])
        u_seq = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['u_dim']])
        mask = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], self.opts['mask_dim']])

        loss_gt = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], None])
        loss_recons = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['nsteps'], None])

        re_loss = recons_loss(self.opts['recons_cost'], loss_gt, loss_recons)

        nll, kl_dist, u_kl_divergence, u_accuracy = self.neg_elbo(x_seq, a_seq, r_seq, u_seq,
                                                                  anneal=self.opts['anneal'], mask=mask)
        x_recons, a_recons, r_recons = self.recons_xar_seq_g_xar_seq(x_seq, a_seq, r_seq, mask)

        train_sample_batch_ids = np.random.choice(data.train_num, self.opts['batch_size'], replace=False)

        train_op = tf.train.AdamOptimizer(self.opts['lr']).minimize(nll)

        print('starting initializing variables ...')

        if self.opts['is_restored']:
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(all_vars, max_to_keep=50)
            self.saver.restore(self.sess, self.opts['model_checkpoint'])
        else:
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=50)

        print('starting epoch ...')

        for epoch in xrange(self.opts['epoch_start'], self.opts['epoch_start'] + self.opts['epoch_num']):

            if epoch > self.opts['epoch_start'] and epoch % self.opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_decon'),
                                global_step=counter)

            ids_perm = np.random.permutation(data.train_num)

            for itr in xrange(batch_num):
                start_time = time.time()

                batch_ids = ids_perm[self.opts['batch_size'] * itr:self.opts['batch_size'] * (itr + 1)]

                _, nll_tr = \
                    self.sess.run([train_op, nll],
                                  feed_dict={x_seq: data.x_train[batch_ids],
                                             a_seq: data.a_train[batch_ids],
                                             r_seq: data.r_train[batch_ids],
                                             u_seq: data.rich_train[batch_ids],
                                             mask: data.mask_train[batch_ids]})

                ####################### training ###################################################################

                x_recons_tr, a_recons_tr, r_recons_tr = \
                    self.sess.run([x_recons, a_recons, r_recons],
                                  feed_dict={x_seq: data.x_train[train_sample_batch_ids],
                                             a_seq: data.a_train[train_sample_batch_ids],
                                             r_seq: data.r_train[train_sample_batch_ids],
                                             u_seq: data.rich_train[train_sample_batch_ids],
                                             mask: data.mask_train[train_sample_batch_ids]})

                x_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.x_train[train_sample_batch_ids],
                                                              loss_recons: x_recons_tr})
                a_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.a_train[train_sample_batch_ids],
                                                              loss_recons: a_recons_tr})
                r_tr_loss = self.sess.run(re_loss, feed_dict={loss_gt: data.r_train[train_sample_batch_ids],
                                                              loss_recons: r_recons_tr})

                elapsed_time = time.time() - start_time

                print('epoch: {:d}, itr: {:d}, nll_tr: {:f}, x_tr_loss: {:f}, a_tr_loss: {:f}, r_tr_loss: {:f}, '
                      'elapsed_time: {:f};'.format(epoch, itr, nll_tr, x_tr_loss, a_tr_loss, r_tr_loss, elapsed_time))

        self.saver.save(self.sess,
                        os.path.join(self.opts['work_dir'], 'model_checkpoints', 'model_decon'),
                        global_step=counter)

