# SAC_tfv2

# Requirements

- pybullet
- tensorflow v2.5
- ray
- gym
- tensorflow_probability

# Usage

```
python SAC.py
```

# Config
You can change config.py to fit your own flag.

```
# used for clipping output of log_std layer
LOG_STD_MIN
LOG_STD_MAX

# Entropy regularization coefficient
alpha_pi
alpha_q


lr          # learning rate
gamma       # discout factor 
polyak      # interpolation factor in polyak averaging
epsilon     # TODO

hdims       # dimension of hidden layers

# ray
n_cpu = n_workers # TODO  cpu num

# learning config
batch_size
update_count
buffer_size_long
buffer_size_short
start_steps

# for loop config
total_steps
evaluate_every
ep_len_rollout
num_eval
max_ep_len_eval
```
