import torch.optim as optim


class Optimizer(object):
    def __init__(self, parameters, config):
        self.config = config
        self.optimizer = build_optimizer(parameters, config)
        self.global_step = 1
        self.current_epoch = 0
        self.lr = config.lr
        self.decay_ratio = config.decay_ratio
        self.epoch_decay_flag = False

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def epoch(self):
        self.current_epoch += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def decay_lr(self):
        self.lr *= self.decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


def build_optimizer(parameters, config):

    if config.type == "adadelta":
        optimizer = optim.Adadelta(parameters, rho=0.95, eps=config.eps, weight_decay=config.weight_decay)

    elif config.type == "adam":
        optimizer = optim.Adam(parameters, weight_decay=config.weight_decay)

    else:
        raise NotImplementedError("unknown optimizer: " + config.opt)

    return optimizer
