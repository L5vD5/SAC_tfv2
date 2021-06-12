buffer_size = 1e6

LOG_STD_MIN = -10.0
LOG_STD_MAX = +2.0

alpha_pi = 0.1
alpha_q = 0.1 #0.1

lr = 1e-3
gamma = 0.98
polyak = 0.995
epsilon = 1e-7

hdims = [256,256]

n_cpu = n_workers = 8
total_steps = 1e6
start_steps = 1e4
evaluate_every = 1e4
ep_len_rollout = 100
batch_size = 128
update_count = 2
num_eval = 3
max_ep_len_eval = 1e3
buffer_size_long = 1e6
buffer_size_short = 1e5