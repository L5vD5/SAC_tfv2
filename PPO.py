import datetime, gym, os, pybullet_envs, psutil, time, os
import scipy.signal
import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
from Replaybuffer import PPOBuffer
from model import *
import random
from config import Config

print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

class Agent(object):
    def __init__(self):
        # Config
        self.config = Config()

        # Environment
        self.env, self.eval_env = get_envs()
        odim = self.env.observation_space.shape[0]
        adim = self.env.action_space.shape[0]

        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.actor_critic = ActorCritic(odim, adim, self.config.hdims,**ac_kwargs)
        self.buf = PPOBuffer(odim=odim,adim=adim,size=self.config.steps_per_epoch,
                             gamma=self.config.gamma,lam=self.config.lam)

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=self.config.pi_lr, epsilon=self.config.epsilon)
        self.train_v = tf.keras.optimizers.Adam(learning_rate=self.config.vf_lr, epsilon=self.config.epsilon)

    @tf.function
    def update_ppo(self, obs, act, adv, ret, logp):
        # self.actor_critic.train()
        # obs = tf.constant(obs)
        # act = tf.constant(act)
        # adv = tf.constant(adv)
        # ret = tf.constant(ret)
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
