import torch

mpsDevice = torch.device("mps")
print(f"Total GPU memory allocated by Metal driver for the process: {torch.mps.driver_allocated_memory()} bytes")

x = torch.rand((10,10), device=mpsDevice)
print(x)
print(f"Total GPU memory allocated by Metal driver for the process: {torch.mps.driver_allocated_memory()} bytes")

y = (x**2)
print(y)
print(f"Total GPU memory allocated by Metal driver for the process: {torch.mps.driver_allocated_memory()} bytes")

print(f"Current GPU memory occupied by tensors: {torch.mps.current_allocated_memory()} bytes")

'''
    Total numbers in tensor x: 9
    Same with tensor y: 9
    Total numbers = 18
'''