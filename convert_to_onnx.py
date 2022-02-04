import torch

from talkNet import talkNet

net = talkNet()
net.loadParameters("pretrain_TalkSet.model")

dummy_input = torch.randn(10, 3, 224, 224)
torch.onnx.export(net.model, dummy_input, f="talknet.onnx", verbose=True, export_params=True, )
