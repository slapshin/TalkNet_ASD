import torch

device_code = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(device_code)

print("detected device: {0}".format(device_code))
