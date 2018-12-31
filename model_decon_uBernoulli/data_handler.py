#########################
## Author: Chaochao Lu ##
#########################
import numpy as np
import logging

class DataHandler(object):
    def __init__(self, opts):
        self.train_num = None
        self.validation_num = None
        self.test_num = None
        self.x_train = None
        self.a_train = None
        self.r_train = None
        self.mask_train = None
        self.x_validation = None
        self.a_validation = None
        self.r_validation = None
        self.mask_validation = None
        self.x_test = None
        self.a_test = None
        self.r_test = None
        self.mask_test = None
        self.train_r_max = None
        self.train_r_min = None
        self.load_data(opts)

    def load_data(self, opts):
        if opts['dataset'] == 'dataset_name':
            self.load_dataset(opts)
        else:
            logging.error(opts['dataset'] + ' cannot be found.')


    def load_dataset(self, opts):
        mnist_train = np.load(opts['training_data'])
        self.x_train = mnist_train['x_train']
        self.a_train = mnist_train['a_train']
        self.r_train = mnist_train['r_train']
        self.mask_train = mnist_train['mask_train']

        mnist_validation = np.load(opts['validation_data'])
        self.x_validation = mnist_validation['x_validation']
        self.a_validation = mnist_validation['a_validation']
        self.r_validation = mnist_validation['r_validation']
        self.mask_validation = mnist_validation['mask_validation']

        mnist_test = np.load(opts['testing_data'])
        self.x_test = mnist_test['x_test']
        self.a_test = mnist_test['a_test']
        self.r_test = mnist_test['r_test']
        self.mask_test = mnist_test['mask_test']

        self.train_num = self.x_train.shape[0]
        self.validation_num = self.x_validation.shape[0]
        self.test_num = self.x_test.shape[0]

        self.train_r_max = np.amax(self.r_train)
        self.train_r_min = np.amin(self.r_train)
