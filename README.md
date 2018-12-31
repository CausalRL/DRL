# Deconfounding Reinforcement Learning (DRL)
This repository contains a clean version of the code for the Deconfounding Reinforcement Learning (DRL) 
model as developed at [1]. The code is easy to modify for your own applications. It is
worth noting that this is my first work within the charming but mysterious kingdom of Causal Reinforcement
Learning (CausalRL). Undoubtedly, CausalRL will become an indispensable part of
Artificial General Intelligence (AGI). Please refer to [CausalRL](http://www.causallu.com/CausalRL.htm) for more details.

## Running Environment
Only tested the codes on 
+ Tesla P100 with 16GB GPU memory 
+ tensorflow-gpu v1.5.1. 

Not sure about other options.

## Data Format
In default the data are organised as follows. Feel free to modify these parameters for your own applications.

                         num    x    nsteps   x      dim
    x_train:           140,000          5            784
    a_train:           140,000          5             1
    r_train:           140,000          5             1
    mask_train:        140,000          5             1
    x_validation:       28,000          5            784
    a_validation:       28,000          5             1
    r_validation:       28,000          5             1
    mask_validation:    28,000          5             1
    x_test:             28,000          5            784
    a_test:             28,000          5             1
    r_test:             28,000          5             1
    mask_test:          28,000          5             1
    
N.B. 1st dim: the number of sequences; 2nd dim: steps of one sequence; 3rd dim: dim of data.

## How to Run
This repo consists of three folders:
+ **model_decon_uGaussian**: learning M_Decon when the dimension of u is set to 2 and the prior over u is assumed to be a factorised standard Gaussian.
  >python run.py
+ **model_decon_uBernoulli**: learning M_Decon when the dimension of u is set to 1 and the prior over u is assumed to be a Bernoulli with p=0.5.
  >python run.py
+ **ac_decon**: learning and testing AC_Decon (In default M_Decon learned in **model_decon_uBernoulli** is required).
  >python train_policy.py
  
  >python test_policy.py
  
Note that before running the code, you are supposed to modify `configs.py`, especially the following lines which involve 
the directories of data and models. 

```python
########################################## data and model path Configuration ###########################################

model_config['work_dir'] = './training_results'
model_config['data_dir'] = './dataset'
model_config['training_data'] = './dataset/training_data.npz'
model_config['validation_data'] = './dataset/validation_data.npz'
model_config['testing_data'] = './dataset/testing_data.npz'
model_config['model_checkpoint'] = './training_results/model_checkpoints/model_alt'
model_config['policy_checkpoint'] = './training_results/policy_checkpoints/policy_alt'

########################################################################################################################
```

##References

[1] [Deconfounding Reinforcement Learning in Observational Settings](Undefined yet)
Chaochao Lu, Bernhard Schölkopf, José Miguel Hernández-Lobato, 2018
