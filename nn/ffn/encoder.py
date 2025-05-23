import math

import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, in_features: int, num_layers: int, out_features: int) -> None:
    super().__init__()
    step_size = (out_features - in_features) / num_layers
    layer_features = [math.ceil(in_features + idx * step_size) for idx in range(num_layers + 1)]
    layer_features[0] = in_features
    layer_features[-1] = out_features
    layers = []
    for idx in range(num_layers):
      layers.append(nn.Linear(layer_features[idx], layer_features[idx + 1]))
      layers.append(nn.ReLU(inplace=True))
    self.encoder_layers = nn.Sequential(*layers)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.encoder_layers(input)
    return out
