import torch

if torch.backends.mps.is_available():
    print("Yes! MPS is available.")
    print(torch.mps.device_count())
else:
    print("No MPS here!")