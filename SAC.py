import datetime, gym, os, pybullet_envs, psutil, time, os
import scipy.signal
import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
import itertools
from Replaybuffer import SACBuffer
from model import *
import random
from config import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

RENDER_ON_EVAL = True

class Agent(object):
    def __init__(self, seed=1):
        self.seed = seed
        # Environment
        self.env, self.eval_env = get_envs()
        odim, adim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Actor-critic model
        self.model = MLPActorCritic(self.odim, self.adim, hdims)
        self.target = MLPActorCritic(self.odim, self.adim, hdims)

        [v_targ.assign(v_main) for v_main, v_targ in zip(self.model.trainable_variables, self.target.trainable_variables)]
        # model load
        # self.model.load_state_dict(tf.load('model_data/model_weights_[64,64]'))
        print("weight load")

        # self.target = deepcopy(self.model)

        # Initialize model
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # parameter chain [q1 + q2]
        # self.q_vars = itertools.chain(self.model.q1.trainable_variables, self.model.q2.trainable_variables)

        self.replay_buffer_long = SACBuffer(odim=odim, adim=adim, size=int(buffer_size_long))
        self.replay_buffer_short = SACBuffer(odim=odim, adim=adim, size=int(buffer_size_short))

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_q = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, o, deterministic=False):
        return self.model.get_action(tf.constant(o.reshape(1, -1)), deterministic)

    # get weihts from model and target layer
    def get_weights(self):
        weight_vals = self.model.state_dict()
        return weight_vals

    def set_weights(self, weight_vals):
        return self.model.load_state_dict(weight_vals)

    @tf.function
    def update_sac(self, replay_buffer):
        pi_loss, var_loss = 0., 0.
        for _ in tf.range(update_count):
            # pi_loss = lambda: self.model.calc_pi_loss(data=replay_buffer)
            with tf.GradientTape() as tape:
                pi_loss = self.model.calc_pi_loss(data=replay_buffer)

            # grad = tape.gradient([pi_loss], self.model.policy.trainable_variables)
            # train_pi_op = self.train_pi.apply_gradients(zip(grad, self.model.policy.trainable_variables))
            train_pi_op = self.train_pi.minimize(pi_loss, var_list=self.model.policy.trainable_variables, tape=tape)

            # var_loss = lambda: self.model.calc_q_loss(target=self.target, data=replay_buffer)
            with tf.control_dependencies([train_pi_op]):
                with tf.GradientTape() as tape:
                    var_loss = self.model.calc_q_loss(target=self.target, data=replay_buffer)
                # grad = tape.gradient([var_loss], self.model.q1.trainable_variables + self.model.q2.trainable_variables)
                # train_q_op = self.train_q.apply_gradients(zip(grad, self.model.q1.trainable_variables + self.model.q2.trainable_variables))
                train_q_op = self.train_q.minimize(var_loss, var_list=self.model.q1.trainable_variables + self.model.q2.trainable_variables, tape=tape)

            with tf.control_dependencies([train_q_op]):
                # Finally, update target networks by polyak averaging.
                for v_main, v_targ in zip(self.model.q1.trainable_variables, self.target.q1.trainable_variables):
                    v_targ.assign(v_main * (1-polyak) + v_targ * polyak)

                for v_main, v_targ in zip(self.model.q2.trainable_variables, self.target.q2.trainable_variables):
                    v_targ.assign(v_main * (1-polyak) + v_targ * polyak)

                for v_main, v_targ in zip(self.model.policy.trainable_variables, self.target.policy.trainable_variables):
                    v_targ.assign(v_main * (1-polyak) + v_targ * polyak)

        return pi_loss, var_loss

    def train(self):
        start_time = time.time()
        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0
        for epoch in range(int(total_steps)):
            if epoch > start_steps:
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
            if epoch >= start_steps:
                for _ in range(int(update_count)):
                    batch = self.replay_buffer_long.sample_batch(batch_size//2)
                    batch_short = self.replay_buffer_short.sample_batch(batch_size//2)

                    batch = {k: tf.constant(v) for k, v in batch.items()}
                    batch_short = {k: tf.constant(v) for k, v in batch_short.items()}

                    replay_buffer = dict(obs1=tf.concat([batch['obs1'], batch_short['obs1']], 0),
                                         obs2=tf.concat([batch['obs2'], batch_short['obs2']], 0),
                                         acts=tf.concat([batch['acts'], batch_short['acts']], 0),
                                         rews=tf.concat([batch['rews'], batch_short['rews']], 0),
                                         done=tf.concat([batch['done'], batch_short['done']], 0)
                                         )
                    pi_loss, var_loss = self.update_sac(replay_buffer)

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % evaluate_every) == 0) or (epoch == (total_steps - 1)):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Evaluate] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, total_steps, epoch / total_steps * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                ep_ret_list = []  # for visualization
                for eval_idx in range(num_eval):
                    o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                    if RENDER_ON_EVAL:
                        _ = self.eval_env.render(mode='human')
                    while not (d or (ep_len == max_ep_len_eval)):
                        a = self.get_action(o, deterministic=False)
                        o, r, d, _ = self.eval_env.step(a.numpy()[0])
                        if RENDER_ON_EVAL:
                            _ = self.eval_env.render(mode='human')
                        ep_ret += r  # compute return
                        ep_len += 1
                        ep_ret_list.append(ep_ret)  # for visualization
                    print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                          % (eval_idx, num_eval, ep_ret, ep_len))


def get_envs():
    env_name = 'AntBulletEnv-v0'
    env,eval_env = gym.make(env_name),gym.make(env_name)
    _ = eval_env.render(mode='human') # enable rendering on test_env
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env

a = Agent()
a.train()
