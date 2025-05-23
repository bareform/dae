from nn.ffn.encoder import Encoder

import torch

def test_encoder_dimensions(model: torch.nn.Module) -> bool:
  input = torch.flatten(torch.randn(1, 1, 28, 28))
  try:
    model(input)
  except:
    return False
  return True

def main() -> bool:
  test_cases = {
    "ENCODER DIMENSION ALIGNMENT": test_encoder_dimensions
  }
  model = Encoder(in_features=784, num_layers=30, out_features=128)
  results = []
  print("Testing Encoder implementation...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_func in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    test_case_status = test_case_func(model)
    if test_case_status:
      print(f"Dimensions of Encoder are aligned")
    else:
      print(f"\033[91mDimensions of Encoder are misaligned\033[0m")
    results.append(test_case_status)
  if all(results):
    print("Encoder implementation is dimension aligned!")
  return all(results)

if __name__ == "__main__":
  main()
