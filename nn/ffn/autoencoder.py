from .encoder import Encoder
from .decoder import Decoder

import torch
import torch.nn as nn

class FeedForwardAutoEncoder(nn.Module):
  def __init__(
    self,
    in_features: int,
    num_encoder_layers: int,
    latent_features: int,
    out_features: int,
    num_decoder_layers: int
  ) -> None:
    super().__init__()
    self.encoder = Encoder(
      in_features=in_features,
      num_layers=num_encoder_layers,
      out_features=latent_features
    )
    self.decoder = Decoder(
      in_features=latent_features,
      num_layers=num_decoder_layers,
      out_features=out_features
    )

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.encoder(input)
    out = self.decoder(out)
    return out
