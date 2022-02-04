import torch

device_code = "gpu" if torch.cuda.is_available() else "cpu"

device = torch.device(device_code)
