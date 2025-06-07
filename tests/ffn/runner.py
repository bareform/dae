from .test_encoder_dimensions import (
  main as test_encoder_dimensions_main
)
from .test_decoder_dimensions import (
  main as test_decoder_dimensions_main
)
from .test_autoencoder_dimensions import (
  main as test_autoencoder_dimensions_main
)

def main():
  runner = {
    "test_encoder_dimensions": test_encoder_dimensions_main,
    "test_decoder_dimensions": test_decoder_dimensions_main,
    "test_autoencoder_dimensions": test_autoencoder_dimensions_main
  }
  results = []
  for test_case_file_main_fn in runner.values():
    test_case_file_main_result = test_case_file_main_fn()
    results.append(test_case_file_main_result)
    print()
  if all(results):
    print("Passed all test cases!")
  else:
    print("FAILED")
    print("Failed the following test cases:")
    for test_case_idx, test_case_file in enumerate(runner.keys()):
      if not results[test_case_idx]:
        print(f"  - {test_case_file}")

if __name__ == "__main__":
  main()
