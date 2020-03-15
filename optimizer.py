import torch

class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, factor, warmup_step, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup_step
        self.factor = factor

    def __str__(self):
        return "learning rate: {} \n warm up step: {}\noptimizer: {}".format(self.factor, self.warmup, self.optimizer)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        if self._step <= self.warmup:
            lr = float(self.factor) * float(self._step) / float(self.warmup)
            for p in self.optimizer.param_groups:
                p['lr'] = lr
        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "optimizer": self.optimizer.state_dict()
        }
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model, warmup, factor):
    base = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    return NoamOpt(factor, warmup, base)
