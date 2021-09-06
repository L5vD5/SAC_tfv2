# SAC_tfv2

## Requirements

- pybullet
- tensorflow v2.5
- ray
- gym
- tensorflow_probability

## Usage

```
$ python main.py --hdims 128 128                             # train without ray
$ python main.py --ray [number_of_workers] --hdims 128 128   # train with ray
$ python main.py --play [weight_file_path] --hdims 128 128   # play with weight file

$ python SAC.py --help

usage: main.py [-h] [--ray RAY] [--play PLAY] [--alpha_pi ALPHA_PI]
               [--alpha_q ALPHA_Q] [--lr LR] [--gamma GAMMA] [--polyak POLYAK]
               --hdims HDIMS [HDIMS ...] [--total_steps TOTAL_STEPS]
               [--start_steps START_STEPS] [--evaluate_every EVALUATE_EVERY]
               [--batch_size BATCH_SIZE] [--update_count UPDATE_COUNT]
               [--num_eval NUM_EVAL] [--max_ep_len_eval MAX_EP_LEN_EVAL]
               [--buffer_size_long BUFFER_SIZE_LONG]
               [--buffer_size_short BUFFER_SIZE_SHORT]

SAC

optional arguments:
  -h, --help            show this help message and exit
  --ray RAY             Train agent with given environment with ray
  --play PLAY           Play agent with given environment
  --alpha_pi ALPHA_PI   Entropy regularization coefficient
  --alpha_q ALPHA_Q     Entropy regularization coefficient
  --lr LR               learning rate
  --gamma GAMMA         discount factor
  --polyak POLYAK       interpolation factor in polyak averaging
  --hdims HDIMS [HDIMS ...]
                        size of hidden dimension
  --total_steps TOTAL_STEPS
                        Number of epochs of interaction (equivalent to number
                        of policy updates) to perform
  --start_steps START_STEPS
                        Number of epochs of interaction (equivalent to number
                        of policy updates) to perform
  --evaluate_every EVALUATE_EVERY
                        How often evaluate be
  --batch_size BATCH_SIZE
                        How big batch size
  --update_count UPDATE_COUNT
                        How many update be on each epoch
  --num_eval NUM_EVAL   How many rollout be on each evaluate
  --max_ep_len_eval MAX_EP_LEN_EVAL
                        Maximum length of trajectory
  --buffer_size_long BUFFER_SIZE_LONG
                        How big long batch size be
  --buffer_size_short BUFFER_SIZE_SHORT
                        How big short batch size be
```

