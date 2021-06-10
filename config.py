# Configuration, set model parameter
class Config:
    def __init__(self):
        # Model
        self.hdims = [32, 32]
        #Graph
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.epsilon = 1e-8
        #Buffer
        self.gamma = 0.99
        self.lam = 0.95
        #Update
        self.train_pi_iters = 100
        self.train_v_iters = 100
        self.target_kl = 0.01
        self.epochs = 5000
        self.max_ep_len = 1000
        self.steps_per_epoch = 1000
        #Worker
        self.print_every = 10
        self.evaluate_every = 50
        self.update_every = 10
