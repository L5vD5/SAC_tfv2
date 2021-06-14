import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box, Discrete
from config import *

##### Model construction #####
def mlp(odim=24, hdims=[64,64], actv='relu', output_actv=None):
    ki = tf.keras.initializers.truncated_normal(stddev=0.1)
    layers = tf.keras.Sequential()
    layers.add(tf.keras.layers.InputLayer(input_shape=(odim,)))
    for hdim in hdims[:-1]:
        layers.add(tf.keras.layers.Dense(hdim, activation=actv, kernel_initializer=ki))
    layers.add(tf.keras.layers.Dense(hdims[-1], activation=output_actv, kernel_initializer=ki))
    return layers

class MLPGaussianPolicy(tf.keras.Model):    # def mlp_gaussian_policy
    def __init__(self, odim, adim, hdims=[64,64], actv='relu'):
        super(MLPGaussianPolicy, self).__init__()
        self.net = mlp(odim, hdims, actv, output_actv=actv) #feature
        # mu layer
        self.mu = tf.keras.layers.Dense(adim, activation=None, trainable=True)
        # std layer
        self.log_std = tf.keras.layers.Dense(adim, activation=None, trainable=True)

    @tf.function
    def call(self, o, deterministic=False, get_logprob=True):
        net_ouput = self.net(o)
        mu = self.mu(net_ouput)
        log_std = self.log_std(net_ouput)

        LOG_STD_MIN, LOG_STD_MAX = -10.0, +2.0
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX) #log_std
        std = tf.exp(log_std) # std

        # Pre-squash distribution and sample
        dist = tfp.distributions.Normal(mu, std)
        if deterministic:
            pi = mu
        else:
            pi = dist.sample()    # sampled

        if get_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = tf.reduce_sum(dist.log_prob(pi), axis=-1)    #gaussian log_likelihood # modified axis
            logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
        else:
            logp_pi = None
        pi = tf.tanh(pi)
        return pi, logp_pi


# Q-function mlp
class MLPQFunction(tf.keras.Model):
    def __init__(self, odim, adim, hdims=[64, 64], actv='relu'):
        super().__init__()
        self.q = mlp(odim+adim, hdims=hdims+[1], actv=actv, output_actv=None)

    @tf.function
    def call(self, o, a):
        x = tf.concat([o, a], -1)
        q = self.q(x)
        return tf.squeeze(q, axis=1)   #Critical to ensure q has right shape.


class MLPActorCritic(tf.keras.Model):   # def mlp_actor_critic
    def __init__(self, odim, adim, hdims=[64,64], actv='relu',
                 output_actv=None, policy=None, action_space=None):
        super(MLPActorCritic,self).__init__()
        self.policy = MLPGaussianPolicy(odim=odim, adim=adim, hdims=hdims, actv=actv)
        self.q1 = MLPQFunction(odim=odim, adim=adim, hdims=hdims, actv=actv)
        self.q2 = MLPQFunction(odim=odim, adim=adim, hdims=hdims, actv=actv)

    @tf.function
    def get_action(self, o, deterministic=False):
        pi, _ = self.policy(o, deterministic, False)
        return pi

    @tf.function
    def calc_pi_loss(self, data):
        o = data['obs1']
        pi, logp_pi = self.policy(o)
        q1_pi = self.q1(o,pi)
        q2_pi = self.q2(o,pi)
        min_q_pi = tf.minimum(q1_pi, q2_pi)
        # tf.print('logp_pi', logp_pi)
        # pi losses
        pi_loss = tf.reduce_mean(alpha_pi*logp_pi - min_q_pi)
        return pi_loss

    @tf.function
    def calc_q_loss(self, target, data):
        o, a, r, o2, d = data['obs1'], data['acts'], data['rews'], data['obs2'], data['done']

        # Entropy-regularized Bellman backup
        # get target action from current policy
        pi_next, logp_pi_next = self.policy(o2)
        # Target value
        q1_targ = target.q1(o2, pi_next)
        q2_targ = target.q2(o2, pi_next)
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        # Entropy-regularized Bellman backup
        q_backup = tf.stop_gradient(r + gamma*(1 - d)*(min_q_targ - alpha_q*logp_pi_next))
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        # tf.print('q1', q1, 'q2', q2)
        # value(q) loss
        q1_loss = 0.5*tf.losses.mse(q1,q_backup)           #0.5 * ((q_backup-q1)**2).mean()
        q2_loss = 0.5*tf.losses.mse(q2,q_backup)          #0.5 * ((q_backup-q2)**2).mean()
        value_loss = q1_loss + q2_loss
        return value_loss