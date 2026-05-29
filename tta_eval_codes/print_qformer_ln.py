import torch
import torch.nn as nn
from evaluate_tta import QformerWrapper

model = QformerWrapper(device='cpu')
for name, module in model.model.Qformer.named_modules():
    if isinstance(module, nn.LayerNorm):
        print(name)
