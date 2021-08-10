import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
from copy import deepcopy
from Replaybuffer import SACBuffer
from model import *
import random
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

RENDER_ON_EVAL = True

class Agent(object):
    def __init__(self, args, seed=1):
        self.seed = seed
        # Environment
        self.env, self.eval_env = get_envs()

        self.total_steps = args.total_steps
        self.start_steps = args.start_steps
        self.evaluate_every = args.evaluate_every
        self.batch_size = args.batch_size
        self.update_count = args.update_count
        self.num_eval = args.num_eval
        self.max_ep_len_eval = args.max_ep_len_eval
        self.buffer_size_long = args.buffer_size_long
        self.buffer_size_short = args.buffer_size_short
        self.polyak = args.polyak

        odim, adim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Actor-critic model
        self.model = MLPActorCritic(self.odim, self.adim, args)
        self.target = MLPActorCritic(self.odim, self.adim, args)

        self.target.set_weights(self.model.get_weights())

        # model load
        # self.model.load_state_dict(tf.load('model_data/model_weights_[64,64]'))
        # print("weight load")

        # Initialize model
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.replay_buffer_long = SACBuffer(odim=odim, adim=adim, size=int(args.buffer_size_long))
        self.replay_buffer_short = SACBuffer(odim=odim, adim=adim, size=int(args.buffer_size_short))

        self.pi_loss_metric = tf.keras.metrics.Mean(name="pi_loss")
        self.value_loss_metric = tf.keras.metrics.Mean(name="Q_loss")
        self.q1_metric = tf.keras.metrics.Mean(name="Q1")
        self.q2_metric = tf.keras.metrics.Mean(name="Q2")
        self.log_path = "./log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")


    def get_action(self, o, deterministic=False):
        return self.model(tf.constant(o.reshape(1, -1)), deterministic)

    # get weihts from model and target layer
    def get_weights(self):
        weight_vals = self.model.state_dict()
        return weight_vals

    def set_weights(self, weight_vals):
        return self.model.load_state_dict(weight_vals)

    @tf.function
    def update_sac(self, replay_buffer):
        # with tf.GradientTape() as tape:
        #     pi_loss = self.model.calc_pi_loss(data=replay_buffer)
        # pi_loss = lambda: self.model.calc_pi_loss(data=replay_buffer)
        # grad = tape.gradient([pi_loss], self.model.policy.trainable_variables)
        # train_pi_op = self.train_pi.apply_gradients(zip(grad, self.model.policy.trainable_variables))
        # self.train_pi.minimize(pi_loss, var_list=self.model.policy.trainable_variables)

        # var_loss = lambda: self.model.calc_q_loss(target=self.target, data=replay_buffer)
        # with tf.GradientTape() as tape:
        #     var_loss = self.model.calc_q_loss(target=self.target, data=replay_buffer)
        # grad = tape.gradient([var_loss], self.model.q1.trainable_variables + self.model.q2.trainable_variables)
        # train_q_op = self.train_q.apply_gradients(zip(grad, self.model.q1.trainable_variables + self.model.q2.trainable_variables))
        # self.train_q.minimize(var_loss, var_list=self.model.q1.trainable_variables + self.model.q2.trainable_variables)
        pi_loss, logp_pi, min_q_pi = self.model.update_policy(replay_buffer)
        value_loss, q1, q2, logp_pi_next, q_backup, q1_targ, q2_targ = self.model.update_Q(self.target, replay_buffer)
        # pi_loss, value_loss, q1, q2 = self.model.update_draft(self.target, replay_buffer)

        self.pi_loss_metric.update_state(pi_loss)
        self.value_loss_metric.update_state(value_loss)
        self.q1_metric.update_state(q1)
        self.q2_metric.update_state(q2)
        # Finally, update target networks by polyak averaging.
        for v_main, v_targ in zip(self.model.q1.trainable_variables, self.target.q1.trainable_variables):
            v_targ.assign(v_main * (1-self.polyak) + v_targ * self.polyak)

        for v_main, v_targ in zip(self.model.q2.trainable_variables, self.target.q2.trainable_variables):
            v_targ.assign(v_main * (1-self.polyak) + v_targ * self.polyak)

        # for v_main, v_targ in zip(self.model.policy.trainable_variables, self.target.policy.trainable_variables):
        #     v_targ.assign(v_main * (1-polyak) + v_targ * polyak)

        return logp_pi, min_q_pi, logp_pi_next, q_backup, q1_targ, q2_targ

    def train(self, load_dir=None):
        start_time = time.time()
        latests_100_score = deque(maxlen=100)
        if load_dir:
            loaded_ckpt = tf.train.latest_checkpoint(load_dir)
            self.model.load_weights(loaded_ckpt)
            print('load weights')

        [v_targ.assign(v_main) for v_main, v_targ in zip(self.model.trainable_variables, self.target.trainable_variables)]

        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0
        for epoch in range(int(self.total_steps)):
            if epoch > self.start_steps:
                a = self.get_action(o, deterministic=False)
                a = a.numpy()[0]
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            ep_len += 1
            ep_ret += r

            # Save the Experience to our buffer
            self.replay_buffer_long.store(o, a, r, o2, d)
            self.replay_buffer_short.store(o, a, r, o2, d)
            n_env_step += 1
            o = o2

            # End of trajectory handling - reset env
            if d:
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform SAC update!
            if epoch >= self.start_steps:
                for _ in tf.range(self.update_count):
                    batch = self.replay_buffer_long.sample_batch(self.batch_size//2)
                    batch_short = self.replay_buffer_short.sample_batch(self.batch_size//2)

                    batch = {k: tf.constant(v) for k, v in batch.items()}
                    batch_short = {k: tf.constant(v) for k, v in batch_short.items()}

                    replay_buffer = dict(obs1=tf.concat([batch['obs1'], batch_short['obs1']], 0),
                                            obs2=tf.concat([batch['obs2'], batch_short['obs2']], 0),
                                            acts=tf.concat([batch['acts'], batch_short['acts']], 0),
                                            rews=tf.concat([batch['rews'], batch_short['rews']], 0),
                                            done=tf.concat([batch['done'], batch_short['done']], 0))
                    logp_pi, min_q_pi, logp_pi_next, q_backup, q1_targ, q2_targ = self.update_sac(replay_buffer)

            # Evaluate
            if (((epoch + 1) % self.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Evaluate] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.total_steps, epoch / self.total_steps * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                for eval_idx in range(self.num_eval):
                    o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                    if RENDER_ON_EVAL:
                        _ = self.eval_env.render(mode='human')
                    while not (d or (ep_len == self.max_ep_len_eval)):
                        a = self.get_action(o, deterministic=True)
                        o, r, d, _ = self.eval_env.step(a.numpy()[0])
                        if RENDER_ON_EVAL:
                            _ = self.eval_env.render(mode='human')
                        ep_ret += r  # compute return
                        ep_len += 1
                    print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                          % (eval_idx, self.num_eval, ep_ret, ep_len))
                latests_100_score.append(ep_ret)
                self.write_summary(epoch, latests_100_score, ep_ret, n_env_step)
                print("Saving weights...")
                self.model.save_weights(self.log_path + "/weights/weights")

    def write_summary(self, episode, latest_100_score, episode_score, total_step):

        with self.summary_writer.as_default():
            tf.summary.scalar("Reward (clipped)", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg reward (clipped)", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("Q1", self.q1_metric.result(), step=episode)
            tf.summary.scalar("Q2", self.q2_metric.result(), step=episode)
            tf.summary.scalar("Value_Loss", self.value_loss_metric.result(), step=episode)
            tf.summary.scalar("PI_Loss", self.pi_loss_metric.result(), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)

        self.q1_metric.reset_states()
        self.q2_metric.reset_states()
        self.value_loss_metric.reset_states()
        self.pi_loss_metric.reset_states()

    def play(self, load_dir=None, trial=5):

        if load_dir:
            loaded_ckpt = tf.train.latest_checkpoint(load_dir)
            self.model.load_weights(loaded_ckpt)

        for i in range(trial):
            o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
            if RENDER_ON_EVAL:
                _ = self.eval_env.render(mode='human')
            while not (d or (ep_len == self.max_ep_len_eval)):
                a = self.get_action(o, deterministic=True)
                o, r, d, _ = self.eval_env.step(a.numpy()[0])
                if RENDER_ON_EVAL:
                    _ = self.eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
            print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (i, self.num_eval, ep_ret, ep_len))

def get_envs():
    env_name = 'AntBulletEnv-v0'
    env,eval_env = gym.make(env_name),gym.make(env_name)
    if RENDER_ON_EVAL:
        _ = eval_env.render(mode='human') # enable rendering on test_env
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env