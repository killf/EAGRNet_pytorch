from torch.optim.lr_scheduler import StepLR, MultiStepLR

from .linear_lr import LinearLR

SCHEDULERS = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "LinearLR": LinearLR
}
