import torch


class RunningStats:

    def __init__(self):
        self.n = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = torch.clone(x)
            self.new_m = torch.clone(x)
            self.old_s = torch.zeros_like(x)
        else:
            delta = x - self.old_m
            self.new_m = self.old_m + delta / self.n
            self.new_s = self.old_s + delta * (x - self.new_m)

            self.old_m = torch.clone(self.new_m)
            self.old_s = torch.clone(self.new_s)

    def mean(self):
        return torch.clone(self.new_m) if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return torch.sqrt(self.variance())
