import datetime, gym, os, pybullet_envs, psutil, time, os
import scipy.signal
import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
from Replaybuffer import SACBuffer
from model import *
import random
from config import *
from collections import deque
# from util import gpu_sess,suppress_tf_warning
np.set_printoptions(precision=2)
# suppress_tf_warning() # suppress warning
gym.logger.set_level(40) # gym logger
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

RENDER_ON_EVAL = False

class RolloutWorkerClass(object):
    """
    Worker without RAY (for update purposes)
    """
    def __init__(self, seed=1):
        self.seed = seed
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        # Actor-critic model
        self.model = MLPActorCritic(self.odim, self.adim, hdims)
        self.target = tf.keras.models.clone_model(self.model)

        # Initialize model
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_v = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def get_action(self, o, deterministic=False):
        return self.model(tf.constant(o.reshape(1, -1)), deterministic)

    @tf.function
    def get_weights(self):
        """
        Get weights
        """
        weight_vals = self.model.trainable_weights
        return weight_vals

    @tf.function
    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        for old_weight, new_weight  in zip(self.model.trainable_weights, weight_vals):
            old_weight.assign(new_weight)

    def save_weight(self, log_path):
        self.model.save_weights(log_path + "/weights/weights")

    def load_weight(self, checkpoint):
        self.model.load_weights(checkpoint)

@ray.remote
class RayRolloutWorkerClass(object):
    """
    Rollout Worker with RAY
    """
    def __init__(self,worker_id=0, ep_len_rollout=1000):
        self.worker_id = worker_id
        self.ep_len_rollout = ep_len_rollout
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.buf = SACBuffer(odim=odim,adim=adim,
                             size=self.config.ep_len_rollout, gamma=self.config.gamma,lam=self.config.lam)

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=self.config.pi_lr, epsilon=self.config.epsilon)
        self.train_v = tf.keras.optimizers.Adam(learning_rate=self.config.vf_lr, epsilon=self.config.epsilon)

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        for idx, weight in enumerate(self.model.trainable_weights):
            weight.assign(weight_vals[idx])

    def rollout(self):
        """
        Rollout
        """
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset()  # reset environment

        o = self.env.reset()
        for t in range(self.config.ep_len_rollout):
            a, _, logp_t, v_t, _ = self.model(o.reshape(1, -1))

            o2, r, d, _ = self.env.step(a.numpy()[0])
            # Save the Experience to our buffer
            self.buf.store(o, a, r, v_t, logp_t)
            o = o2
            if d:
                self.buf.finish_path(last_val=0.0)
                self.o = self.env.reset()  # reset when done
        last_val = self.model.vf_mlp(tf.constant(o.reshape(1, -1))).numpy()[0][0]
        self.buf.finish_path(last_val)
        return self.buf.get()

class Agent(object):
    def __init__(self):
        # Config
        # self.config = Config()

        # Environment
        self.eval_env = get_env()
        odim = self.eval_env.observation_space.shape[0]
        adim = self.eval_env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        ray.init(num_cpus=n_cpu)
        self.R = RolloutWorkerClass(seed=0)
        self.workers = [RayRolloutWorkerClass.remote(worker_id=i, ep_len_rollout=self.config.ep_len_rollout)
                   for i in range(int(n_cpu))]
        print("RAY initialized with [%d] cpus and [%d] workers." %
              (n_cpu, n_workers))

        self.replay_buffer_long = SACBuffer(odim=odim, adim=adim, size=int(buffer_size_long))
        self.replay_buffer_short = SACBuffer(odim=odim, adim=adim, size=int(buffer_size_short))

        self.pi_loss_metric = tf.keras.metrics.Mean(name="pi_loss")
        self.v_loss_metric = tf.keras.metrics.Mean(name="V_loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q")
        self.log_path = "./log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")

    def train(self):
        latests_100_score = deque(maxlen=100)
        start_time = time.time()
        n_env_step = 0  # number of environment steps

        for epoch in range(total_steps):
            # 1. Synchronize worker weights
            weights = self.R.get_weights()
            set_weights_list = [worker.set_weights.remote(weights) for worker in self.workers]
            # 2. Make rollout and accumulate to Buffers
            t_start = time.time()
            ops = [worker.rollout.remote() for worker in self.workers]
            rollout_vals = ray.get(ops)
            sec_rollout = time.time() - t_start
            # 3. Update
            t_start = time.time()  # tic

            # obs_bufs, act_bufs, adv_bufs, ret_bufs, logp_bufs =
            # Perform PPO update!
            for r_idx, rval in enumerate(rollout_vals):
                obs_buf, act_buf, adv_buf, ret_buf, logp_buf = \
                    rval[0], rval[1], rval[2], rval[3], rval[4]
                if r_idx == 0:
                    obs_bufs, act_bufs, adv_bufs, ret_bufs, logp_bufs = \
                        obs_buf, act_buf, adv_buf, ret_buf, logp_buf
                else:
                    obs_bufs = np.concatenate((obs_bufs, obs_buf), axis=0)
                    act_bufs = np.concatenate((act_bufs, act_buf), axis=0)
                    adv_bufs = np.concatenate((adv_bufs, adv_buf), axis=0)
                    ret_bufs = np.concatenate((ret_bufs, ret_buf), axis=0)
                    logp_bufs = np.concatenate((logp_bufs, logp_buf), axis=0)

            sec_update = time.time() - t_start  # toc

            # Print
            if (epoch == 0) or (((epoch + 1) % print_every) == 0):
                print("[%d/%d] rollout:[%.1f] pi_iter:[%d/%d] update:[%.1f] kl:[%.4f] target_kl:[%.4f]." %
                      (epoch + 1, epochs, sec_rollout, pi_iter, train_pi_iters, sec_update, kl, target_kl))
                print("pi_loss:[%.4f], v_loss:[%.4f], entropy:[%.4f]" % (pi_loss, v_loss, ent))

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, epochs, epoch / epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == max_ep_len)):
                    a, _, _, _ = self.R.model.policy(tf.constant(o.reshape(1, -1)))
                    o, r, d, _ = self.eval_env.step(a.numpy()[0])
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))
                latests_100_score.append((ep_ret))
                self.write_summary(epoch, latests_100_score, ep_ret, n_env_step)
                print("Saving weights...")
                self.R.model.save_weights(self.log_path + "/weights/weights")

    def write_summary(self, episode, latest_100_score, episode_score, total_step):

        with self.summary_writer.as_default():
            tf.summary.scalar("Reward (clipped)", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg reward (clipped)", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("V_Loss", self.v_loss_metric.result(), step=episode)
            tf.summary.scalar("PI_Loss", self.pi_loss_metric.result(), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)

        self.v_loss_metric.reset_states()
        self.pi_loss_metric.reset_states()
        # self.q_metric.reset_states()


def get_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    eval_env = gym.make('AntBulletEnv-v0')
    if RENDER_ON_EVAL:
        _ = eval_env.render(mode='human') # enable rendering
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return eval_env

a = Agent()
a.train()