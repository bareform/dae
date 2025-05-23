from nn.ffn import FeedForwardAutoEncoder

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg") # run matplotlib in 'headless' mode
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def get_argparser():
  parser = argparse.ArgumentParser(prog="trainer",
                      description="training loop for AutoEncoder")
  parser.add_argument("--root", type=str, default=os.path.join("data", "datasets"))
  parser.add_argument("--dataset", type=str, default="MNIST",
                      choices=["MNIST", "FashionMNIST"])
  parser.add_argument("--batch_size", type=int, default=512)
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--pin_memory", action="store_true", default=True)
  parser.add_argument("--num_epochs", type=int, default=5000)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--optimizer", type=str, default="AdamW",
                      choices=["AdamW"])
  parser.add_argument("--lr_scheduler", type=str, default="StepLR",
                      choices=["StepLR"])
  parser.add_argument("--step_size", type=int, default=200)
  parser.add_argument("--gamma", type=float, default=0.9)
  parser.add_argument("--model_type", type=str, default="ffn",
                      choices=["ffn"])
  parser.add_argument("--num_encoder_layers", type=int, default=3)
  parser.add_argument("--num_decoder_layers", type=int, default=3)
  parser.add_argument("--latent_features", type=int, default=128)
  parser.add_argument("--random_seed", type=int, default=0)
  parser.add_argument("--compile", action="store_true", default=False)
  parser.add_argument("--ckpt_dir", type=str, default=os.path.join("checkpoints"))
  parser.add_argument("--results_dir", type=str, default=os.path.join("results"))
  parser.add_argument("--save_interval", type=int, default=3000)
  return parser

def main():
  args = get_argparser().parse_args()
  pad_length = len(str(args.num_epochs))

  torch.manual_seed(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  if not (os.path.exists(args.ckpt_dir) and os.path.isdir(args.ckpt_dir)):
    os.makedirs(args.ckpt_dir, exist_ok=True)

  results_dir = os.path.join(args.results_dir, args.dataset, args.model_type, "epoch")
  if not (os.path.exists(results_dir) and os.path.isdir(results_dir)):
    os.makedirs(results_dir, exist_ok=True)

  gif_dir = os.path.join(args.results_dir, args.dataset, args.model_type, "gif")
  if not (os.path.exists(gif_dir) and os.path.isdir(gif_dir)):
    os.makedirs(gif_dir, exist_ok=True)

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
  ])

  print(f"Training on {args.dataset}")
  if args.dataset == "MNIST":
    train_set = torchvision.datasets.MNIST(root=args.root, train=True, transform=transform)
    train_loader = data.DataLoader(
      train_set,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory
    )
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
    test_noise = torch.randn_like(test_images)
    test_noise = test_noise.to(device)
    noisy_test_images = test_images + test_noise
  elif args.dataset == "FashionMNIST":
    train_set = torchvision.datasets.FashionMNIST(root=args.root, train=True, transform=transform)
    train_loader = data.DataLoader(
      train_set,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory
    )
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
    test_noise = torch.randn_like(test_images)
    test_noise = test_noise.to(device)
    noisy_test_images = test_images + test_noise
  else:
    raise RuntimeError("something went wrong: dataset was not set!")

  print("Training: FeedForward AutoEncoder")
  dummy_image, _ = train_set[0]
  dummy_image = dummy_image.flatten()
  in_features = dummy_image.numel()
  if args.model_type == "ffn":
    model = FeedForwardAutoEncoder(
      in_features=in_features,
      num_encoder_layers=args.num_encoder_layers,
      latent_features=args.latent_features,
      num_decoder_layers=args.num_decoder_layers,
      out_features=in_features
    )
  else:
    raise RuntimeError("use 'ffn' as 'model_type'")
  model = model.to(device)
  if args.compile:
    print("Compiling with Just-In-Time compilation")
    model = torch.compile(model)
  criterion = torch.nn.MSELoss()
  if args.optimizer == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
  else:
    raise RuntimeError("something went wrong: optimizer was not set!")
  if args.lr_scheduler == "StepLR":
    scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=args.step_size,
      gamma=args.gamma
    )
  else:
    raise RuntimeError("something went wrong: learning rate scheduler was not set!")

  model.train()
  for epoch in range(args.num_epochs):
    running_loss = 0.0
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
      for images, _ in train_loader:
        images = images.to(device)
        noise = torch.randn_like(images)
        noise = noise.to(device)
        noisy_images = images + noise
        optimizer.zero_grad()
        output = model(noisy_images.view(noisy_images.size(0), -1))
        loss = criterion(output, images.view(images.size(0), -1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({
          "loss": f"{loss.item():.5f}"
        })
    train_loss = running_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}/{args.num_epochs}")
    print(f"Train Loss: {train_loss:.5f}")
    scheduler.step()
    if (epoch + 1) % args.save_interval == 0:
      print("Saving checkpoint model")
      torch.save(
        {
          "in_features": in_features,
          "num_encoder_layers": args.num_encoder_layers,
          "latent_features": args.latent_features,
          "num_decoder_layers": args.num_decoder_layers,
          "model_state_dict": model.state_dict(),
          "model_type": args.model_type,
          "compile": args.compile,
          "dataset": args.dataset
        },
        os.path.join(args.ckpt_dir, f"{args.dataset}_{args.model_type}_epoch{epoch + 1:0{pad_length}d}.pth")
      )

    with torch.no_grad():
      denoised_test_images = model(noisy_test_images.view(noisy_test_images.size(0), -1))
      denoised_test_images = denoised_test_images.view(denoised_test_images.size(0), 1, test_height, test_width)
    combined_images = torch.cat([noisy_test_images, denoised_test_images, test_images], dim=0)
    grid_img = torchvision.utils.make_grid(combined_images, nrow=12)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(
      results_dir, f"{epoch + 1:0{pad_length}d}.png"
    ), bbox_inches="tight", pad_inches=0)
    plt.close()

  print("Saving gif")
  output_gif = os.path.join(args.results_dir, args.dataset, args.model_type, "gif", f"denoising.gif")
  png_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".png")])
  images = [Image.open(os.path.join(results_dir, f)) for f in png_files]
  images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=150,
    loop=0
  )

if __name__ == "__main__":
  main()
