from nn.ffn.decoder import Decoder

import torch

def test_decoder_dimensions(model: torch.nn.Module) -> bool:
  input = torch.flatten(torch.randn(1, 1, 28, 28))
  try:
    model(input)
  except:
    return False
  return True

def main() -> bool:
  test_cases = {
    "DECODER DIMENSION ALIGNMENT": test_decoder_dimensions
  }
  model = Decoder(in_features=784, num_layers=30, out_features=128)
  results = []
  print("Testing Decoder implementation...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_func in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    test_case_status = test_case_func(model)
    if test_case_status:
      print(f"Dimensions of Decoder are aligned")
    else:
      print(f"\033[91mDimensions of Decoder are misaligned\033[0m")
    results.append(test_case_status)
  if all(results):
    print("Decoder implementation is dimension aligned!")
  return all(results)

if __name__ == "__main__":
  main()
