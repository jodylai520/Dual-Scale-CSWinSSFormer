from trainner import Trainer_synapse
from config import MSUNet_synapse_version4_cls9_v1 as hyper4
from thop import profile, clever_format

import torch

net = Trainer_synapse("/home/lthpc/joe/Synapse_npy", hyper4)


# def print_model_size():
#     x = torch.randn((2, 1, 224, 224)).to('cpu')
#     model = net.model.to('cpu')
#     flops, params = profile(model=model, inputs=(x,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print(flops, params)
