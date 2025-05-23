from nn.ffn.encoder import Encoder

import torch

def test_num_layers_six_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 6:
    print("Saw six nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected six torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 6:
    print("Saw six nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected six torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  alternating_linear_and_relu = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(12)
  ]
  has_alternating_linear_and_relu = True
  for encoder_layer, expected_layer in zip(model.encoder_layers, alternating_linear_and_relu):
    if not isinstance(encoder_layer, expected_layer):
      has_alternating_linear_and_relu = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_alternating_linear_and_relu:
    print("Encoder has alternating torch.nn.Linear layers and torch.nn.ReLU layers")
  else:
    print("\033[91mEncoder does not have alternating torch.nn.Linear layers and torch.nn.ReLU layers\033[0m")
  return all(test_case_status)

def test_num_layers_five_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 5:
    print("Saw five nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected five torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 5:
    print("Saw five nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected five torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  alternating_linear_and_relu = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(10)
  ]
  has_alternating_linear_and_relu = True
  for encoder_layer, expected_layer in zip(model.encoder_layers, alternating_linear_and_relu):
    if not isinstance(encoder_layer, expected_layer):
      has_alternating_linear_and_relu = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_alternating_linear_and_relu:
    print("Encoder has alternating torch.nn.Linear layers and torch.nn.ReLU layers")
  else:
    print("\033[91mEncoder does not have alternating torch.nn.Linear layers and torch.nn.ReLU layers\033[0m")
  return all(test_case_status)

def test_num_layers_three_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 3:
    print("Saw three nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected three torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 3:
    print("Saw three nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected three torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  alternating_linear_and_relu = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(6)
  ]
  has_alternating_linear_and_relu = True
  for encoder_layer, expected_layer in zip(model.encoder_layers, alternating_linear_and_relu):
    if not isinstance(encoder_layer, expected_layer):
      has_alternating_linear_and_relu = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_alternating_linear_and_relu:
    print("Encoder has alternating torch.nn.Linear layers and torch.nn.ReLU layers")
  else:
    print("\033[91mEncoder does not have alternating torch.nn.Linear layers and torch.nn.ReLU layers\033[0m")
  return all(test_case_status)

def test_num_layers_two_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 2:
    print("Saw two nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected two torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 2:
    print("Saw two nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected two torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  alternating_linear_and_relu = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(2*2)
  ]
  has_alternating_linear_and_relu = True
  for encoder_layer, expected_layer in zip(model.encoder_layers, alternating_linear_and_relu):
    if not isinstance(encoder_layer, expected_layer):
      has_alternating_linear_and_relu = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_alternating_linear_and_relu:
    print("Encoder has alternating torch.nn.Linear layers and torch.nn.ReLU layers")
  else:
    print("\033[91mEncoder does not have alternating torch.nn.Linear layers and torch.nn.ReLU layers\033[0m")
  return all(test_case_status)

def test_num_layers_one_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 1:
    print("Saw one nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected one torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.encoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 1:
    print("Saw one nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected one torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  alternating_linear_and_relu = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(2)
  ]
  has_alternating_linear_and_relu = True
  for encoder_layer, expected_layer in zip(model.encoder_layers, alternating_linear_and_relu):
    if not isinstance(encoder_layer, expected_layer):
      has_alternating_linear_and_relu = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_alternating_linear_and_relu:
    print("Encoder has alternating torch.nn.Linear layers and torch.nn.ReLU layers")
  else:
    print("\033[91mEncoder does not have alternating torch.nn.Linear layers and torch.nn.ReLU layers\033[0m")
  return all(test_case_status)

def main() -> bool:
  test_cases = {
    "SIX LAYERS": (Encoder(in_features=784, num_layers=6, out_features=128), test_num_layers_six_layers),
    "FIVE LAYERS": (Encoder(in_features=784, num_layers=5, out_features=128), test_num_layers_five_layers),
    "THREE LAYERS": (Encoder(in_features=784, num_layers=3, out_features=128), test_num_layers_three_layers),
    "TWO LAYERS": (Encoder(in_features=784, num_layers=2, out_features=128), test_num_layers_two_layers),
    "ONE LAYERS": (Encoder(in_features=784, num_layers=1, out_features=128), test_num_layers_one_layers)
  }
  results = []
  print("Testing Encoder implemenation...")
  print("DETAILED REPORT")
  print("---------------")
  for test_case_name, test_case_model_func_pair in test_cases.items():
    print(f"Problem: {test_case_name}")
    print("Feedback")
    model, test_case_func = test_case_model_func_pair
    test_case_status = test_case_func(model)
    if test_case_status:
      print("PASSED")
    else:
      print("FAILED")
    print("---------------")
    results.append(test_case_status)
  if all(results):
    print("Encoder model is setup correctly!")
  else:
    print("FAILED")
    print("Failed the following test cases:")
    for test_case_idx, test_case_name in enumerate(test_cases.keys()):
      if not results[test_case_idx]:
        print(f"  - {test_case_name}")
    return False
  return all(results)

if __name__ == "__main__":
  main()
