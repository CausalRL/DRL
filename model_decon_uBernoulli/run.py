#########################
## Author: Chaochao Lu ##
#########################
import configs
import os
import tensorflow as tf
from model_decon import Model_Decon
from data_handler import DataHandler


def main():
    opts = configs.model_config

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    print('starting processing data ...')

    data = DataHandler(opts)


    print('starting initialising model ...')
    opts['r_range_upper'] = data.train_r_max
    opts['r_range_lower'] = data.train_r_min
    model = Model_Decon(sess, opts)

    print('starting training model ...')
    model.train_model(data)


if __name__ == "__main__":
    main()

