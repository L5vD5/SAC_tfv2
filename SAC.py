import datetime, gym, os, pybullet_envs, psutil, time, os
import scipy.signal
import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
from copy import deepcopy
from Replaybuffer import SACBuffer
from model import *
import itertools
import random
from config import *

print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

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

        # model load
        # self.model.load_state_dict(tf.load('model_data/model_weights_[64,64]'))
        print("weight load")

        self.target = deepcopy(self.model)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False

        # Initialize model
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # parameter chain [q1 + q2]
        self.q_vars = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())

        replay_buffer_long = SACBuffer(odim=odim, adim=adim, size=int(buffer_size_long))
        replay_buffer_short = SACBuffer(odim=odim, adim=adim, size=int(buffer_size_short))

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_v = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, o, deterministic=False):
        return self.model.get_action(tf.constant(o.reshape(1, -1)), deterministic)

    # get weihts from model and target layer
    def get_weights(self):
        weight_vals = self.model.state_dict()
        return weight_vals

    def set_weights(self, weight_vals):
        return self.model.load_state_dict(weight_vals)

    @tf.function
    def update_ppo(self, obs, act, adv, ret, logp):
        logp_a_old = logp

        for _ in tf.range(self.config.train_pi_iters):

            with tf.GradientTape() as tape:
                # pi, logp, logp_pi, mu
                _, logp_a, _, _ = self.actor_critic.policy(obs, act)
                ratio = tf.exp(logp_a - logp_a_old)  # pi(a|s) / pi_old(a|s)
                min_adv = tf.where(adv > 0, (1 + self.config.clip_ratio) * adv, (1 - self.config.clip_ratio) * adv)
                pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

            gradients = tape.gradient(pi_loss, self.actor_critic.policy.trainable_weights)
            self.train_pi.apply_gradients(zip(gradients, self.actor_critic.policy.trainable_variables))

            # _, logp_a, _, _ = self.actor_critic.policy(obs, act)
            # ratio = tf.exp(logp_a - logp_a_old)  # pi(a|s) / pi_old(a|s)
            # min_adv = tf.where(adv > 0, (1 + self.config.clip_ratio) * adv, (1 - self.config.clip_ratio) * adv)
            # pi_loss = lambda: -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))
            #
            # self.train_pi.minimize(pi_loss, var_list=[self.actor_critic.policy.trainable_variables])

            kl = tf.reduce_mean(logp_a_old - logp_a)
            if kl > 1.5 * self.config.target_kl:
                break

        for _ in tf.range(self.config.train_v_iters):
            with tf.GradientTape() as tape:
                v = tf.squeeze(self.actor_critic.vf_mlp(obs))
                v_loss = tf.keras.losses.MSE(v, ret)

            gradients = tape.gradient(v_loss, self.actor_critic.vf_mlp.trainable_weights)
            self.train_v.apply_gradients(zip(gradients, self.actor_critic.vf_mlp.trainable_variables))

    def train(self):
        start_time = time.time()
        o, r, d, ep_ret, ep_len, n_env_step = self.eval_env.reset(), 0, False, 0, 0, 0

        for epoch in range(self.config.epochs):
            if (epoch == 0) or (((epoch + 1) % self.config.print_every) == 0):
                print("[%d/%d]" % (epoch + 1, self.config.epochs))
            o = self.env.reset()
            for t in range(self.config.steps_per_epoch):
                a, _, logp_t, v_t, _ = self.actor_critic(o.reshape(1, -1))

                o2, r, d, _ = self.env.step(a.numpy()[0])
                ep_ret += r
                ep_len += 1
                n_env_step += 1

                # Save the Experience to our buffer
                self.buf.store(o, a, r, v_t, logp_t)
                o = o2

                terminal = d or (ep_len == self.config.max_ep_len)
                if terminal or (t == (self.config.steps_per_epoch - 1)):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = 0 if d else self.actor_critic.vf_mlp(tf.constant(o.reshape(1, -1))).numpy()[0][0]
                    self.buf.finish_path(last_val)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform PPO update!
            obs, act, adv, ret, logp = [tf.constant(x) for x in self.buf.get()]
            self.update_ppo(obs, act, adv, ret, logp)

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % self.config.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.config.epochs, epoch / self.config.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.config.max_ep_len)):
                    a, _, _, _ = self.actor_critic.policy(tf.constant(o.reshape(1, -1)))
                    o, r, d, _ = self.eval_env.step(a.numpy()[0])
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

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
