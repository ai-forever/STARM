import copy

import torch.nn as nn


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

        self.extra_buffer_patterns = ["puzzle_emb.weights"]

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        for name, buffer in module.named_buffers():
            if any(pattern in name for pattern in self.extra_buffer_patterns):
                self.shadow[name] = buffer.data.clone()
                print(f"  ✓ EMA tracking buffer: {name}")

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

        for name, buffer in module.named_buffers():
            if name in self.shadow:
                self.shadow[name].data = (1. - self.mu) * buffer.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

        for name, buffer in module.named_buffers():
            if name in self.shadow:
                buffer.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
