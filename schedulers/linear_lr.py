class LinearLR:
    def __init__(self, optimizer, learning_rates, milestones, last_epoch=-1):
        assert len(learning_rates) == len(milestones) + 1

        self.optimizer = optimizer
        self.learning_rates = learning_rates
        self.milestones = [1] + milestones
        self.last_epoch = last_epoch
        self._last_lr = None

    def state_dict(self):
        return {"learning_rates": self.learning_rates, "milestones": self.milestones, "last_epoch": self.last_epoch, "_last_lr": self._last_lr}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.learning_rates = state_dict["learning_rates"]
        self.milestones = state_dict["milestones"]
        self.last_epoch = state_dict["last_epoch"]
        self._last_lr = state_dict["_last_lr"]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        if self.last_epoch <= 1:
            return self.learning_rates[0]
        elif self.last_epoch >= self.milestones[-1]:
            return self.learning_rates[-1]

        for i in range(len(self.milestones) - 1):
            if self.milestones[i] <= self.last_epoch <= self.milestones[i + 1]:
                if self.milestones[i] == self.milestones[i + 1]:
                    return self.learning_rates[i]

                k = (self.learning_rates[i + 1] - self.learning_rates[i]) / (self.milestones[i + 1] - self.milestones[i])
                lr = k * (self.last_epoch - self.milestones[i]) + self.learning_rates[i]
                return lr

        return self.learning_rates[-1]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            lr = self.get_lr()
        else:
            self.last_epoch = epoch
            lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self._last_lr = lr
