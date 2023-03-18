import torch

from model import UNet as Unet
from utils import SynapseLoss

NCLS = 9
MS_synapse_version1_cls9_v10_1 = {
    'describe': "optim hyper",
    'save_path': "E:\\result\\MS\\version10\\v1",
    'dataset_path': "E:\\Synapse_data",
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'epoch_num': 800,
    'num_classes': NCLS,
    'pretrained_params': None,
    'model': Unet,
    'model_args': {'img_size': 256, 'in_chans': 1, 'embed_dim': 64, 'num_classes': NCLS},
    'criterion': SynapseLoss,
    'criterion_args': {'n_classes': NCLS, 'alpha': 0.1, 'beta': 0.9},
    # number of classes should equal to the number of out channels in model_args
    'optimizer': torch.optim.AdamW,
    'optimizer_args': {'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
    'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
    'scheduler_args': {'T_max': 90000},
    'train_loader_args': {'batch_size': 28, 'shuffle': True, 'num_workers': 8, 'pin_memory': True, 'drop_last': True},
    'test_loader_args': {'batch_size': 1, 'shuffle': False, 'num_workers': 1},
    'eval_frequncy': 30,  # test/inference frequncy
    'save_frequncy': 5,  # save model state dict frequncy
    'n_gpu': 1,
    'grad_clipping': False,
}
