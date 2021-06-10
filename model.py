import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box, Discrete

##### Model construction #####
def mlp(odim=24, hdims=[256,256], actv='relu', output_actv=None):
    layers = tf.keras.Sequential()
    layers.add(tf.keras.layers.InputLayer(input_shape=(odim,)))
    for hdim in hdims[:-1]:
        layers.add(tf.keras.layers.Dense(hdim, activation=actv))
    layers.add(tf.keras.layers.Dense(hdims[-1], activation=output_actv))
    return layers

class CategoricalPolicy(tf.keras.Model):
    def __init__(self, odim, adim, hdims=[64,64], actv='relu', output_actv=None):
        super(CategoricalPolicy, self).__init__()
        self.net = mlp(odim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.logits = tf.keras.layers.Dense(adim)
    def __call__(self, x, a=None):
        output = self.net(x)
        logits = self.logits(output)
        prob = tf.nn.softmax(logits)
        dist = tfp.distributions.Categorical(probs=prob)
        pi = dist.sample()
        logp_pi = dist.log_prob(pi)
        logp = dist.log_prob(a)
        return pi, logp, logp_pi, pi

class GaussianPolicy(tf.keras.Model):    # def mlp_gaussian_policy
    def __init__(self, odim, adim, hdims=[64,64], actv='relu', output_actv=None):
        super(GaussianPolicy, self).__init__()
        self.mu = mlp(odim, hdims=hdims+[adim], actv=actv, output_actv=output_actv)
        self.log_std = tf.Variable(-0.5*tf.ones(adim))
    @tf.function
    def call(self, x, a=None):
        mu = self.mu(x)
        std = tf.math.exp(self.log_std)
        policy = tfp.distributions.Normal(mu, std)
        pi = policy.sample()
        # gaussian likelihood
        logp_pi = tf.reduce_sum(policy.log_prob(pi), axis=1)
        if a is not None:
            logp = tf.reduce_sum(policy.log_prob(a), axis=1)
        else:
            logp = None
        return pi, logp, logp_pi, mu        # 순서 ActorCritic return 값이랑 맞춤.


class ActorCritic(tf.keras.Model):   # def mlp_actor_critic
    def __init__(self, odim, adim, hdims=[64,64], actv='relu',
                 output_actv=None, policy=None, action_space=None):
        super(ActorCritic,self).__init__()
        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(odim, adim, hdims, actv, output_actv)
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(odim, adim, hdims, actv, output_actv)
        self.vf_mlp = mlp(odim, hdims=hdims+[1],
                          actv=actv, output_actv=output_actv)
    @tf.function
    def call(self, x, a=None):
        pi, logp, logp_pi, mu = self.policy(x, a)
        v = self.vf_mlp(x)
        return pi, logp, logp_pi, v, mu
