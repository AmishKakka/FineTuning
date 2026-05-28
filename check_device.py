import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Memory allocated: ", torch.mps.recommended_max_memory())
        print("MPS available (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU")
    return device