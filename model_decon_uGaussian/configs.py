#########################
## Author: Chaochao Lu ##
#########################
from collections import OrderedDict
import tensorflow as tf

########################################################################################################################
########################################## Full Model Configuration ####################################################
########################################################################################################################

model_config = OrderedDict()

########################################## data and model path Configuration ###########################################

model_config['work_dir'] = './training_results'
model_config['data_dir'] = './dataset'
model_config['training_data'] = './dataset/training_data.npz'
model_config['validation_data'] = './dataset/validation_data.npz'
model_config['testing_data'] = './dataset/testing_data.npz'

########################################################################################################################

model_config['dataset'] = 'dataset_name'
model_config['seed'] = 123
model_config['lr'] = 0.0001

model_config['is_conv'] = True
model_config['gated'] = True

model_config['is_restored'] = False
model_config['model_checkpoint'] = None
model_config['epoch_start'] = 0
model_config['counter_start'] = 0

model_config['init_std'] = 0.0099999
model_config['init_bias'] = 0.0
model_config['filter_size'] = 5

model_config['a_range'] = 2

model_config['z_dim'] = 50
model_config['x_dim'] = 784  # 28 x 28
model_config['a_dim'] = 1
model_config['u_dim'] = 2
model_config['a_latent_dim'] = 100
model_config['r_dim'] = 1
model_config['r_latent_dim'] = 100
model_config['mask_dim'] = 1
model_config['lstm_dim'] = 100
model_config['mnist_dim'] = 28
model_config['mnist_channel'] = 1

model_config['batch_size'] = 128
model_config['nsteps'] = 5
model_config['sample_num'] = 5
model_config['epoch_num'] = 400

model_config['save_every_epoch'] = 10
model_config['plot_every'] = 500
model_config['inference_model_type'] = 'LR'
model_config['lstm_dropout_prob'] = 0.
model_config['recons_cost'] = 'l2sq'
model_config['anneal'] = 1

model_config['work_dir'] = './training_results'
model_config['data_dir'] = '/scratch/cl641/CausalRL/bcdr_beta/data_prep'

model_config['pxgz_net_layers'] = [100, 100]
model_config['pxgz_net_outlayers'] = []
model_config['pxgu_net_layers'] = [100, 100]
model_config['pxgu_net_outlayers'] = []
model_config['pxgzu_prenet_layers'] = [512]
model_config['pxgzu_prenet_outlayers'] = []
model_config['pxgzu_in_shape'] = [[7, 7, 32], [14, 14, 16]]
model_config['pxgzu_out_shape'] = [[[28, 28, 1], tf.nn.sigmoid],
                                  [[28, 28, 1], tf.nn.softplus]]

model_config['pxgzu_net_layers'] = [300, 300]
model_config['pxgzu_net_outlayers'] = [[model_config['x_dim'], tf.nn.sigmoid],
                                      [model_config['x_dim'], tf.nn.softplus]]


model_config['pagz_net_layers'] = [100, 100]
model_config['pagz_net_outlayers'] = []

model_config['pagu_net_layers'] = [100, 100]
model_config['pagu_net_outlayers'] = []

model_config['pagzu_net_layers'] = [300, 300]
model_config['pagzu_net_outlayers'] = [[model_config['a_dim'], tf.nn.tanh],
                                       [model_config['a_dim'], tf.nn.softplus]]

model_config['prgz_net_layers'] = [100, 100]
model_config['prgz_net_outlayers'] = []
model_config['prga_net_layers'] = [100, 100]
model_config['prga_net_outlayers'] = []
model_config['prgu_net_layers'] = [100, 100]
model_config['prgu_net_outlayers'] = []
model_config['prgzau_net_layers'] = [300, 300]
model_config['prgzau_net_outlayers'] = [[model_config['r_dim'], tf.nn.sigmoid],
                                      [model_config['r_dim'], tf.nn.softplus]]

model_config['pzgz_net_layers'] = [100]
model_config['pzgz_net_outlayers'] = []
model_config['pzga_net_layers'] = [model_config['a_latent_dim']]
model_config['pzga_net_outlayers'] = []

model_config['pzgza_net_layers'] = [100]
model_config['pzgza_net_outlayers'] = []
model_config['pzgza_mu_net_layers'] = []
model_config['pzgza_mu_net_outlayers'] = [[model_config['z_dim'], None]]

model_config['pzgza_pregate_net_layers'] = [100]
model_config['pzgza_pregate_net_outlayers'] = []
model_config['pzgza_gate_net_layers'] = []
model_config['pzgza_gate_net_outlayers'] = [[model_config['z_dim'], tf.nn.sigmoid]]
model_config['pzgza_gate_mu_net_layers'] = [100]
model_config['pzgza_gate_mu_net_outlayers'] = [[model_config['z_dim'], None]]
model_config['pzgza_sigma_net_layers'] = []
model_config['pzgza_sigma_net_outlayers'] = [[model_config['z_dim'], tf.nn.softplus]]

model_config['qzgx_in_channels'] = [16, 32, 32]
model_config['qzgx_out_channel'] = []
model_config['qzgx_encoded_net_layers'] = [100]
model_config['qzgx_encoded_net_outlayers'] = []

model_config['qzgx_net_layers'] = [300, 300, 100]
model_config['qzgx_net_outlayers'] = []
model_config['qzga_net_layers'] = [100, model_config['a_latent_dim']]
model_config['qzga_net_outlayers'] = []
model_config['qzgr_net_layers'] = [100, model_config['r_latent_dim']]
model_config['qzgr_net_outlayers'] = []
model_config['qzgxar_net_layers'] = [100]
model_config['qzgxar_net_outlayers'] = [[100, None]]
model_config['qagh_net_layers'] = []
model_config['qagh_net_outlayers'] = [[model_config['lstm_dim'], None]]


model_config['qugx_in_channels'] = [16, 32, 32]
model_config['qugx_out_channel'] = []
model_config['qugx_encoded_net_layers'] = [100]
model_config['qugx_encoded_net_outlayers'] = []

model_config['qugx_net_layers'] = [300, 300, 100]
model_config['qugx_net_outlayers'] = []
model_config['quga_net_layers'] = [100, model_config['a_latent_dim']]
model_config['quga_net_outlayers'] = []
model_config['qugr_net_layers'] = [100, model_config['r_latent_dim']]
model_config['qugr_net_outlayers'] = []
model_config['qugxar_net_layers'] = [100]
model_config['qugxar_net_outlayers'] = [[100, None]]

model_config['qugh_net_layers']  = [100]
model_config['qugh_net_outlayers']  = [[model_config['u_dim'], None],
                                       [model_config['u_dim'], tf.nn.softplus]]



model_config['lstm_net_layers'] = []
model_config['lstm_net_outlayers'] = [[model_config['lstm_dim'], None]]

model_config['qagx_in_channels'] = [16, 32, 32]
model_config['qagx_out_channel'] = []
model_config['qagx_encoded_net_layers'] = []
model_config['qagx_encoded_net_outlayers'] = [[model_config['a_dim'], tf.nn.tanh],
                                              [model_config['a_dim'], tf.nn.softplus]]

model_config['qagx_net_layers'] = [300, 300, 100]
model_config['qagx_net_outlayers'] = [[model_config['a_dim'], tf.nn.tanh],
                                      [model_config['a_dim'], tf.nn.softplus]]

model_config['qrgx_in_channels'] = [16, 32, 32]
model_config['qrgx_out_channel'] = []
model_config['qrgx_encoded_net_layers'] = [100]
model_config['qrgx_encoded_net_outlayers'] = []

model_config['qrgx_net_layers'] = [300, 300, 100]
model_config['qrgx_net_outlayers'] = []

model_config['qrga_net_layers'] = [model_config['a_latent_dim']]
model_config['qrga_net_outlayers'] = []
model_config['qrgxa_net_layers'] = [100]
model_config['qrgxa_net_outlayers'] = [[model_config['r_dim'], tf.nn.sigmoid],
                                       [model_config['r_dim'], tf.nn.softplus]]


model_config['model_bn_is_training'] = True

