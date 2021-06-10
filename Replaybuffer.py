import numpy as np
import scipy.signal

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class SACBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, odim, adim, size=5000):
        self.obs1_buf = np.zeros(combined_shape(size, odim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, odim), dtype=np.float32)
        self.acts_buf = np.zeros(combined_shape(size, adim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs1=self.obs1_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: v for k, v in batch.items()}

    def get(self):
        names = ['obs1_buf','obs2_buf','acts_buf','rews_buf','done_buf',
                 'ptr','size','max_size']
        vals =[self.obs1_buf,self.obs2_buf,self.acts_buf,self.rews_buf,self.done_buf,
               self.ptr,self.size,self.max_size]
        return names,vals

    def restore(self,a):
        self.obs1_buf = a[0]
        self.obs2_buf = a[1]
        self.acts_buf = a[2]
        self.rews_buf = a[3]
        self.done_buf = a[4]
        self.ptr = a[5]
        self.size = a[6]
        self.max_size = a[7]
