import torch

from model import UNet as Unet
from utils import SynapseLoss

NCLS = 9
MS_synapse_version1_cls9_v10_2 = {
    'describe': "optim hyper",
    'save_path': "E:\\result\\MS\\version10\\v2",
    'dataset_path': "E:\\Synapse_data",
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    'epoch_num': 300,
    'num_classes': NCLS,
    'pretrained_params': None,
    'model': Unet,
    'model_args': {'img_size': 256, 'in_chans': 1, 'embed_dim': 64, 'num_classes': NCLS},
    'criterion': SynapseLoss,
    'criterion_args': {'n_classes': NCLS, 'alpha': 0.3, 'beta': 0.7},
    # number of classes should equal to the number of out channels in model_args
    'optimizer': torch.optim.AdamW,
    'optimizer_args': {'lr': 0.00008, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
    'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
    'scheduler_args': {'T_max': 20000},
    'train_loader_args': {'batch_size': 14, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': True},
    'test_loader_args': {'batch_size': 1, 'shuffle': False, 'num_workers': 1},
    'eval_frequncy': 20,  # test/inference frequncy
    'save_frequncy': 5,  # save model state dict frequncy
    'n_gpu': 1,
    'grad_clipping': False,
}
