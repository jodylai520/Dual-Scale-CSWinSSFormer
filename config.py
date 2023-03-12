import torch
import os

from model import UNet
from utils import DiceLoss

NCLS = 9
MSUNet_synapse_version4_cls9_v1 = {
    'save_path': '/home/lthpc/daniel/results/MS/MS_V9_3',
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    'epoch_num': 240,
    'num_classes': NCLS,
    'pretrained_params': None,
    'model': UNet,
    'model_args': {'img_size': 256, 'in_chans': 1, 'embed_dim': 64, 'num_classes': NCLS},
    'criterion': DiceLoss,
    'criterion_args': {'n_classes': NCLS},  # number of classes should equal to the number of out channels in model_args
    'optimizer': torch.optim.AdamW,
    'optimizer_args': {'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_args': {},
    'train_loader_args': {'batch_size': 16, 'shuffle': True, 'num_workers': 1, 'pin_memory': True, 'drop_last': True},
    'validate_loader_args': {'batch_size': 16, 'shuffle': True, 'num_workers': 1, 'pin_memory': True, 'drop_last': True},
    'eval_frequncy': 5,
    'flags': {
        'save_history': True
    }
}
