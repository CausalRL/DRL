#########################
## Author: Chaochao Lu ##
#########################
from collections import deque
import time
import random
from utils import *

class AC_Decon(object):

    ####################################################################################################################
    ################################################ construct computational graph #####################################
    ####################################################################################################################

    def __init__(self, sess, opts, model):
        self.sess = sess
        self.opts = opts
        self.model = model
        self.saver = None

        ################################################################################################################

        np.random.seed(0)

        ################################################ make our environment ##########################################

        self.x = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['x_dim']])
        self.z = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['z_dim']])
        self.a = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['a_dim']])
        self.r = tf.placeholder(tf.float32, shape=[self.opts['batch_size'], self.opts['r_dim']])
        self.u = tf.placeholder(tf.float32, shape=[self.opts['u_sample_size'],
                                                   self.opts['batch_size'], self.opts['u_dim']])

        # compute initial hidden state z_0 given x_0, a_0, r_0
        _, z_mu_init, z_cov_init = self.model.q_z_g_z_x_a_r(self.x, self.a, self.r)

        # using re-parameterization trick
        # eps_init = tf.random_normal((self.opts['batch_size'], self.opts['z_dim']), 0., 1., dtype=tf.float32)
        # self.z_init = z_mu_init + tf.multiply(eps_init, tf.sqrt(1e-8 + z_cov_init))
        self.z_init = z_mu_init

        # Case 1: u has a standard Gaussian prior
        # self.u_init = tf.random_normal((self.opts['u_sample_size'], self.opts['batch_size'], self.opts['u_dim']),
        #                                0., 1., dtype=tf.float32)

        # Case 2: u has a Bernoulli prior with p=0.5
        self.u_init = tf.to_float(tf.random_uniform((self.opts['u_sample_size'], self.opts['batch_size'],
                                                     self.opts['u_dim']), 0., 1., dtype=tf.float32) < 0.5)

        # compute z_next given z_current and a_current
        self.z_mu_next, z_cov_next = self.model.p_z_g_z_a(self.z, self.a)
        # using re-parameterization trick
        # eps_z_next = tf.random_normal((self.opts['batch_size'], self.opts['z_dim']), 0., 1., dtype=tf.float32)
        # self.z_next = self.z_mu_next + tf.multiply(eps_z_next, tf.sqrt(1e-8 + z_cov_next))
        self.z_next = self.z_mu_next

        self.reward = None
        self.r_next = self.compute_r_g_cu()

        # load prior policy: copy parameters of p_a_g_z to those of actor_net
        # self.a_prior_mu, self.a_prior_sigma = self.model.p_a_g_z(self.z)
        self.model_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        ############################################ setup for policy ##################################################

        self.z_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.a_ph = tf.placeholder(tf.float32, shape=[None, self.opts['a_dim']])
        self.r_ph = tf.placeholder(tf.float32, shape=[None])
        self.z_next_ph = tf.placeholder(tf.float32, shape=[None, self.opts['z_dim']])
        self.is_not_terminal_ph = tf.placeholder(tf.float32, shape=[None])

        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())

        self.episodes = tf.Variable(0.0, trainable=False)
        self.episode_increase_op = self.episodes.assign_add(1)

        self.replay_memory = deque(maxlen=self.opts['replay_memory_capacity'])

        ################################## construct architecture for ACPG_TD ##########################################

        # construct the actor network with its corresponding target network
        with tf.variable_scope('actor_net'):
            self.a_mu, self.a_sigma = self.actor_net(self.z_ph, False, self.is_training_ph, True)
        # re-parameterization trick
        eps = tf.random_normal((1, self.opts['a_dim']), 0., 1., dtype=tf.float32)
        self.a_samples = tf.clip_by_value(self.a_mu + tf.multiply(eps, tf.sqrt(1e-8 + self.a_sigma)),
                                          -self.opts['a_range'], self.opts['a_range'])

        with tf.variable_scope('target_actor_net', reuse=False):
            self.target_a_mu, self.target_a_sigma = self.actor_net(self.z_next_ph, False, self.is_training_ph, False)
            self.target_a_mu = tf.stop_gradient(self.target_a_mu)
            self.target_a_sigma = tf.stop_gradient(self.target_a_sigma)

        # construct the critic network with its corresponding target network
        with tf.variable_scope('critic_net'):
            self.q_a_mu, self.q_a_sigma = self.critic_net(self.z_ph, self.a_ph, False, self.is_training_ph, True)
            self.q_sa_mu, self.q_sa_sigma = self.critic_net(self.z_ph, self.a_mu, True, self.is_training_ph, True)

        with tf.variable_scope('target_critic_net', reuse=False):
            self.q_next_mu, self.q_next_sigma = \
                self.critic_net(self.z_next_ph, self.target_a_mu, False, self.is_training_ph, False)
            self.q_next_mu = tf.stop_gradient(self.q_next_mu)
            self.q_next_sigma = tf.stop_gradient(self.q_next_sigma)

        # collect vars for each network
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')
        target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_net')

        # construct ops of updating parameters of target networks
        update_target_ops = []
        for i, target_actor_var in enumerate(target_actor_vars):
            update_target_actor_op = target_actor_var.assign(self.opts['tau'] * actor_vars[i] +
                                                             (1 - self.opts['tau']) * target_actor_var)
            update_target_ops.append(update_target_actor_op)

        for i, target_critic_var in enumerate(target_critic_vars):
            update_target_critic_op = target_critic_var.assign(self.opts['tau'] * critic_vars[i] +
                                                               (1 - self.opts['tau']) * target_critic_var)
            update_target_ops.append(update_target_critic_op)

        self.update_targets_op = tf.group(*update_target_ops)

        ######################################### compute loss functions ###############################################

        # one-step temporal difference error
        targets = tf.expand_dims(self.r_ph, 1) + \
                  tf.expand_dims(self.is_not_terminal_ph, 1) * self.opts['gamma'] * self.q_next_mu
        td_errors = targets - self.q_a_mu

        # critic loss function (mean square value error with regularization)
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in critic_vars:
            if 'b' not in var.name:
                critic_loss += self.opts['l2_reg_critic'] * 0.5 * tf.nn.l2_loss(var)

        critic_lr = self.opts['lr_critic'] * self.opts['lr_decay'] ** self.episodes
        self.critic_train_op = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss, var_list=critic_vars)

        # actor loss function (mean Q-values under current policy with regularization)
        actor_loss = -1 * tf.reduce_mean(self.q_sa_mu)

        # additional regularization terms
        actor_loss += gaussianNLL(self.a_ph, self.a_mu, self.a_sigma)

        for var in actor_vars:
            if 'b' not in var.name:
                actor_loss += self.opts['l2_reg_actor'] * 0.5 * tf.nn.l2_loss(var)

        actor_lr = self.opts['lr_actor'] * self.opts['lr_decay'] ** self.episodes
        self.actor_train_op = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list=actor_vars)



    ####################################################################################################################
    ################################################ functions related to env ##########################################
    ####################################################################################################################

    def create_env(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.opts['policy_checkpoint'])


    def compute_z_init(self, x, a, r):
        z_init = self.sess.run(self.z_init, feed_dict={self.x: x, self.a: a, self.r: r})
        return z_init

    def compute_u_init(self, x, a, r):
        u_init = self.sess.run(self.u_init, feed_dict={self.x: x, self.a: a, self.r: r})
        return u_init

    def compute_r_g_cu(self):
        self.reward = []
        for i in xrange(self.opts['u_sample_size']):
            reward_mu, _ = self.model.p_r_g_z_a_u(self.z, self.a,
                                                  tf.reshape(self.u[i], [self.opts['batch_size'],
                                                                         self.opts['u_dim']]))
            self.reward.append(reward_mu)
        return tf.reduce_mean(self.reward)

    def step(self, z, a, x, u):
        # r_next: batch_size x r_dim
        z_next_value, r_next_value, reward_samples = \
            self.sess.run([self.z_next, self.r_next, self.reward],
                          feed_dict={self.z: z, self.a: a, self.x: x, self.u: u})

        done = np.abs(r_next_value - self.opts['final_reward']) == 0


        return z_next_value, np.reshape(r_next_value, self.opts['r_dim'])[0], np.reshape(done, 1)[0], reward_samples


    ####################################################################################################################
    ################################################ functions related to policy #######################################
    ####################################################################################################################

    def actor_net(self, z, reuse, is_training, trainable):
        # policy function
        mu, sigma = ac_fc_net(self.opts, z, self.opts['policy_net_layers'],
                              self.opts['policy_net_outlayers'], 'policy_net',
                              reuse=reuse, is_training=is_training, trainable=trainable)
        mu = mu * 2
        return mu, sigma

    def critic_net(self, z, a, reuse, is_training, trainable):
        # stochastic value function
        z_a = tf.concat([z, a], axis=1)
        mu, sigma = ac_fc_net(self.opts, z_a, self.opts['value_net_layers'],
                              self.opts['value_net_outlayers'], 'value_net',
                              reuse=reuse, is_training=is_training, trainable=trainable)
        return mu, sigma

    def choose_action(self, z, is_training):
        action = self.sess.run(self.a_samples, feed_dict={self.z_ph: z, self.is_training_ph: is_training})
        return action

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

    ####################################################################################################################
    ################################################ training ##########################################################
    ####################################################################################################################

    def policy_test(self, data):

        self.saver = tf.train.Saver(max_to_keep=50)
        total_steps = 0
        total_episode = 0

        reward_list = []
        t1_percent_list = []



        self.create_env()

        for episode in xrange(self.opts['policy_test_episode_num']):

            r_f = open(os.path.join(self.opts['work_dir'], 'plots', 'policy_test_ac_decon_reward_list.txt'), 'a+')
            t_f = open(os.path.join(self.opts['work_dir'], 'plots', 'policy_test_ac_decon_t1_percent_list.txt'), 'a+')


            total_reward = 0
            steps_in_episode = 0

            tr_batch_ids = np.random.choice(data.test_num, self.opts['batch_size'], replace=False)
            tr_nstep_ids = np.random.choice(self.opts['nsteps'], 1)
            tr_x_init = np.reshape(data.x_test[tr_batch_ids][:, tr_nstep_ids, :],
                                   [self.opts['batch_size'], self.opts['x_dim']])
            tr_a_init = np.reshape(data.a_test[tr_batch_ids][:, tr_nstep_ids, :],
                                   [self.opts['batch_size'], self.opts['a_dim']])
            tr_r_init = np.reshape(data.r_test[tr_batch_ids][:, tr_nstep_ids, :],
                                   [self.opts['batch_size'], self.opts['r_dim']])

            z = self.compute_z_init(tr_x_init, tr_a_init, tr_r_init)
            u_est = self.compute_u_init(tr_x_init, tr_a_init, tr_r_init)

            action_list = []
            for step in xrange(self.opts['max_steps_in_episode']):
                action = self.choose_action(z, False)

                if np.abs(np.reshape(action, 1)[0]) >= 1:
                    action_list.append(1)
                else:
                    action_list.append(0)

                z_next, reward, done, reward_samples = self.step(z, action, tr_x_init, u_est)

                total_reward += reward

                z = z_next
                total_steps += 1
                steps_in_episode += 1

                if done:
                    _ = self.sess.run(self.episode_increase_op)
                    break

            total_episode += 1

            print('Episode: {:d}, Steps: {:d}, Reward: {:f}'.format(episode, steps_in_episode, total_reward))

            reward_list.append(total_reward)
            t1_percent_list.append(np.sum(action_list)/(self.opts['max_steps_in_episode']*1.0))

            r_f.write('{:f}\n'.format(total_reward))
            t_f.write('{:f}\n'.format(np.sum(action_list)/(self.opts['max_steps_in_episode']*1.0)))


            r_f.close()
            t_f.close()



