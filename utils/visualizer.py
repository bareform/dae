from nn.ffn import FeedForwardAutoEncoder

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

def get_argparser():
  parser = argparse.ArgumentParser(prog="visualizer",
                      description="visualizer for AutoEncoder")
  parser.add_argument("--model_checkpoint", required=True)
  parser.add_argument("--root", type=str, default=os.path.join("data", "datasets"))
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--pin_memory", action="store_true", default=True)
  parser.add_argument("--random_seed", type=int, default=0)
  parser.add_argument("--results_dir", type=str, default=os.path.join("results"))
  return parser

def main():
  args = get_argparser().parse_args()

  torch.manual_seed(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  try:
    checkpoint = torch.load(args.model_checkpoint)
  except:
    raise RuntimeError("failed to load model checkpoint")

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
  ])

  print(f"Visualizing on {checkpoint['dataset']}")
  if checkpoint["dataset"] == "MNIST":
    test_set = torchvision.datasets.MNIST(root=args.root, train=False, transform=transform)
    test_loader = iter(data.DataLoader(
      test_set,
      batch_size=12,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory
    ))
    test_images, _ = next(test_loader)
    test_height, test_width  = 28, 28
  elif checkpoint["dataset"] == "FashionMNIST":
    test_set = torchvision.datasets.FashionMNIST(root=args.root, train=False, transform=transform)
    test_loader = iter(data.DataLoader(
      test_set,
      batch_size=12,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory
    ))
    test_images, _ = next(test_loader)
    test_height, test_width  = 28, 28
  else:
    raise RuntimeError("unknown dataset")

  state_dict = checkpoint["model_state_dict"]
  new_state_dict = {}
  if checkpoint["compile"]:
    for k in state_dict:
      v = state_dict[k]
      k = k.replace("_orig_mod.", "")
      new_state_dict[k] = v
    state_dict = new_state_dict

  if checkpoint["model_type"] == "ffn":
    model = FeedForwardAutoEncoder(
      in_features=checkpoint["in_features"],
      num_encoder_layers=checkpoint["num_encoder_layers"],
      latent_features=checkpoint["latent_features"],
      num_decoder_layers=checkpoint["num_decoder_layers"],
      out_features=checkpoint["in_features"]
    )
  else:
    raise RuntimeError("unknown model type")
  model = model.to(device)
  model.load_state_dict(state_dict)
  model.eval()

  noise = torch.randn_like(test_images)
  noise = noise.to(device)
  noisy_test_images = test_images + noise

  with torch.no_grad():
    denoised_test_images = model(noisy_test_images.view(noisy_test_images.size(0), -1))
    denoised_test_images = denoised_test_images.view(denoised_test_images.size(0), 1, test_height, test_width)
  combined_images = torch.cat([noisy_test_images, denoised_test_images, test_images], dim=0)

  results_dir = os.path.join("results", checkpoint["dataset"], checkpoint["model_type"], "checkpoint")
  if not (os.path.exists(results_dir) and os.path.isdir(results_dir)):
    os.makedirs(results_dir, exist_ok=True)

  grid_img = torchvision.utils.make_grid(combined_images, nrow=12)
  plt.figure(figsize=(12, 4))
  plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
  plt.axis("off")
  plt.tight_layout()
  plt.savefig(os.path.join(
    results_dir, "checkpoint_denoising_results.png"
  ), bbox_inches="tight", pad_inches=0)
  plt.show()

if __name__ == "__main__":
  main()
