import torch

device_code = "gpu" if torch.cuda.is_available() else "cuda"

device = torch.device(device_code)

print("detected device: {0}".format(device_code))
