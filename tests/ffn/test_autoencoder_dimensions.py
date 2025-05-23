from nn.ffn.autoencoder import FeedForwardAutoEncoder

import torch

def test_autoencoder_dimensions(model: torch.nn.Module) -> bool:
  input = torch.flatten(torch.randn(1, 1, 28, 28))
  try:
    model(input)
  except:
    return False
  return True

def main() -> bool:
  test_cases = {
    "AUTOENCODER DIMENSION ALIGNMENT": test_autoencoder_dimensions
  }
  model = FeedForwardAutoEncoder(
    in_features=784,
    num_encoder_layers=3,
    latent_features=128,
    num_decoder_layers=3,
    out_features=784
  )
  results = []
  print("Testing AutoEncoder implementation...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_func in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    test_case_status = test_case_func(model)
    if test_case_status:
      print(f"Dimensions of AutoEncoder are aligned")
    else:
      print(f"\033[91mDimensions of AutoEncoder are misaligned\033[0m")
    results.append(test_case_status)
  if all(results):
    print("AutoEncoder implementation is dimension aligned!")
  return all(results)

if __name__ == "__main__":
  main()
