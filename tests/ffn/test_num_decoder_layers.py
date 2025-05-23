from nn.ffn.decoder import Decoder

import torch

def test_num_layers_six_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 6:
    print("Saw six nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected six torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 5:
    print("Saw five nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected five torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  pattern = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(11)
  ] + [torch.nn.Sigmoid]
  has_right_pattern = True
  for decoder_layer, expected_layer in zip(model.decoder_layers, pattern):
    if not isinstance(decoder_layer, expected_layer):
      has_right_pattern = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_right_pattern:
    print("Decoder has the correct pattern")
  else:
    print("\033[91mDecoder does not have the correct pattern\033[0m")
  return all(test_case_status)

def test_num_layers_five_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 5:
    print("Saw five nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected five torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 4:
    print("Saw four nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected four torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  pattern = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(9)
  ] + [torch.nn.Sigmoid]
  has_right_pattern = True
  for decoder_layer, expected_layer in zip(model.decoder_layers, pattern):
    if not isinstance(decoder_layer, expected_layer):
      has_right_pattern = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_right_pattern:
    print("Decoder has the correct pattern")
  else:
    print("\033[91mDecoder does not have the correct pattern\033[0m")
  return all(test_case_status)

def test_num_layers_three_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 3:
    print("Saw three nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected three torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 2:
    print("Saw two nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected two torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  pattern = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(5)
  ] + [torch.nn.Sigmoid]
  has_right_pattern = True
  for decoder_layer, expected_layer in zip(model.decoder_layers, pattern):
    if not isinstance(decoder_layer, expected_layer):
      has_right_pattern = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_right_pattern:
    print("Decoder has the correct pattern")
  else:
    print("\033[91mDecoder does not have the correct pattern\033[0m")
  return all(test_case_status)

def test_num_layers_two_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 2:
    print("Saw two nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected two torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 1:
    print("Saw one nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected one torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  pattern = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(3)
  ] + [torch.nn.Sigmoid]
  has_right_pattern = True
  for decoder_layer, expected_layer in zip(model.decoder_layers, pattern):
    if not isinstance(decoder_layer, expected_layer):
      has_right_pattern = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_right_pattern:
    print("Decoder has the correct pattern")
  else:
    print("\033[91mDecoder does not have the correct pattern\033[0m")
  return all(test_case_status)

def test_num_layers_one_layers(model: torch.nn.Module) -> bool:
  test_case_status = []
  num_linear_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.Linear))
  if num_linear_layers == 1:
    print("Saw one nn.Linear layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected one torch.nn.Linear layers\033[0m")
    test_case_status.append(False)
  num_relu_layers = sum(1 for layer in model.decoder_layers if isinstance(layer, torch.nn.ReLU))
  if num_relu_layers == 0:
    print("Saw zero nn.ReLU layers")
    test_case_status.append(True)
  else:
    print(f"\033[91mExpected zero torch.nn.ReLU layers\033[0m")
    test_case_status.append(False)
  pattern = [
    torch.nn.Linear if idx % 2 == 0 else torch.nn.ReLU for idx in range(1)
  ] + [torch.nn.Sigmoid]
  has_right_pattern = True
  for decoder_layer, expected_layer in zip(model.decoder_layers, pattern):
    if not isinstance(decoder_layer, expected_layer):
      has_right_pattern = False
      test_case_status.append(False)
      break
  test_case_status.append(True)
  if has_right_pattern:
    print("Decoder has the correct pattern")
  else:
    print("\033[91mDecoder does not have the correct pattern\033[0m")
  return all(test_case_status)

def main() -> bool:
  test_cases = {
    "SIX LAYERS": (Decoder(in_features=784, num_layers=6, out_features=128), test_num_layers_six_layers),
    "FIVE LAYERS": (Decoder(in_features=784, num_layers=5, out_features=128), test_num_layers_five_layers),
    "THREE LAYERS": (Decoder(in_features=784, num_layers=3, out_features=128), test_num_layers_three_layers),
    "TWO LAYERS": (Decoder(in_features=784, num_layers=2, out_features=128), test_num_layers_two_layers),
    "ONE LAYERS": (Decoder(in_features=784, num_layers=1, out_features=128), test_num_layers_one_layers)
  }
  results = []
  print("Testing Decoder implemenation...")
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
    print("Decoder model is setup correctly!")
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
