# ------------------------------------------
# DAFN: Dual Attention Fusion Network
# Licensed under the MIT License.
# written By Ruixin Yang
# ------------------------------------------

import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
import datetime
import logging
import sys
import json

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from dataset import Synapse_dataset
from dataset import RandomGenerator as RandomTransform
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume

def inference(model, hyper, labels, save_infrence=False, test_mode=False):
   model.eval()
   metric_list = 0.0
   num_classes = hyper['num_classes']
   img_size = hyper['model_args']['img_size']
   test_save_path = None
   if save_infrence:
      test_save_path = os.path.join(hyper['save_path'],'inference')
      if os.path.exists(test_save_path) is False:
         os.makedirs(test_save_path)

   device = torch.device(hyper['device'])
   dataset_path = hyper['dataset_path'] 
   test_loader_args = hyper['test_loader_args'] if 'test_loader_args' in hyper else {}
   test_path = os.path.join(dataset_path,'test_vol_h5')
   test_dataset = Synapse_dataset(base_dir=test_path, split="test_vol", list_dir=dataset_path, img_size=img_size,shuffle=False , test_mode=test_mode)
   test_loader = DataLoader(test_dataset, **test_loader_args)
   length = len(test_loader)

   for i_batch,sample in enumerate(test_loader):
      image, label, case_name = sample['image'], sample['label'], sample['case_name'][0]
      metric_i = test_single_volume(image, label, model, device, classes=num_classes, 
                                    patch_size=[img_size, img_size], test_save_path=test_save_path, case=case_name)
      metric_list += np.array(metric_i)
      metric_mean = np.mean(metric_i,axis=0)
      logging.info(f'\t\t {case_name}: mean dice {metric_mean[0]:.4f}, mean hd95 {metric_mean[1]:.4f}')

   metric_list /= length
   
   for i, organ in enumerate(labels):
            logging.info(f'\t\t {organ} : dice {metric_list[i][0]:.4f}, hd95 {metric_list[i][1]:.4f}')
   

   logging.info(f'\t\t mean dice {np.mean(metric_list,axis=0)[0]:.4f}, mean hd95 {np.mean(metric_list,axis=0)[1]:.4f}')

   return np.asarray(metric_list) #[[dice][hd95]]

def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')

def train_epoch(model, criterion, optimizer,scheduler, hyper, test_mode=False, train_loader=None):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    grad_clipping = hyper['grad_clipping']
    device = torch.device(hyper['device'])

    if train_loader is None:
       dataset_path = hyper['dataset_path']
       img_size = hyper['model_args']['img_size'] if 'img_size' in hyper['model_args'].keys() else 224
       train_loader_args = hyper['train_loader_args'] if 'train_loader_args' in hyper else {}
       x_transforms = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize([0.5], [0.5])
       ])
       y_transforms = transforms.ToTensor()
   
       train_path = os.path.join(dataset_path,'train_npz')
       train_dataset = Synapse_dataset(base_dir=train_path, list_dir=dataset_path, split="train",img_size=img_size,
                                  norm_x_transform = x_transforms, norm_y_transform = y_transforms,shuffle=True,test_mode=test_mode)
       train_loader = DataLoader(train_dataset, **train_loader_args)

    length = len(train_loader)
    for idx, sample in enumerate(train_loader):
      x,y = sample['image'], sample['label']
      x = x.to(device)
      y = y.to(device).squeeze(1)
      
      y_ = model(x)

      loss,ce,dice = criterion(y_, y)
      running_loss += loss
      running_ce += ce
      running_dice += dice

      optimizer.zero_grad()
      loss.backward()
      if grad_clipping:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
      
      optimizer.step()

      if scheduler is not None:
             scheduler.step()
             lr_ = scheduler.get_last_lr()[-1]
      # else:
      #    lr_ = args.base_lr * (1.0 - args.iter_num / args.max_iterations) ** 0.9
      #    for param_group in optimizer.param_groups:
      #       param_group['lr'] = lr_
      
      # args.iter_num += 1
      # logging.info('iteration %d : lr: %f, loss : %f, loss_ce: %f, loss_dice: %f' % (args.iter_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
    
    logging.info(f'\t\t loss:{running_loss/length:.4f}')
    logging.info(f'\t\t ce loss   :{running_ce/length:.4f}')
    logging.info(f'\t\t dice loss :{running_dice/length:.4f}')
    logging.info(f'\t\t lr :{lr_}')
    # return running_loss/length, running_ce/length, running_dice/length, lr_

class Trainer_synapse():
  def __init__(self , hyper):
    super().__init__()
    
    self.hyper = hyper
    self.epoch_num = hyper['epoch_num']
    self.eval_frequncy = hyper['eval_frequncy'] or 5

    #---------- save path configurations ----------
    self.save_path = hyper['save_path']
    if os.path.exists(self.save_path) == False:
       os.makedirs(self.save_path)

    log_path = os.path.join(self.save_path, 'log.txt')
    # if os.path.exists(log_path):
    #    os.remove(log_path)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info('Synapse Trainer Initailizing...')
    logging.info(f'result will be saved on {self.save_path}')
    self.save_epoch_path = os.path.join(self.save_path, 'best_epoch.pth')
    
    if os.path.exists(self.save_epoch_path) or os.path.exists(os.path.join(self.save_path,'epoch.pth')):
       logging.info("========================warning====================")
       logging.info("This config was trained before, continue training will cover the saved data!")
       time.sleep(3)
       logging.info("====================================================")
      # date = time.strftime("%Y%m%d_%H%M",time.localtime(time.time()))
      # os.rename(self.save_epoch_path,os.path.join(self.save_path,f'best_epoch_{date}.pth'))
  
    #---------- setup model ----------
    self.num_classes = hyper['num_classes']
    self.device = torch.device(hyper['device'])
    self.model = hyper['model'](**hyper['model_args']).to(self.device)
    if hyper['pretrained_params'] is not None:
      self.model.load_from(hyper['pretrained_params'], self.device)
      self.model.to(self.device)

    self.criterion = hyper['criterion'](**hyper['criterion_args'])
    self.optimizer = hyper['optimizer'](self.model.parameters(),**hyper['optimizer_args'])
    self.scheduler = hyper['scheduler'](self.optimizer,**hyper['scheduler_args']) if 'scheduler' in hyper else None
    
    #---------- relavent args ----------
    if hyper['n_gpu'] > 1:
        self.model = nn.DataParallel(self.model)

    self.labels = {0: 'background',  1: 'spleen',      2: 'right kidney',
                   3: 'left kidney', 4: 'gallbladder', 5: 'liver',
                   6: 'stomach',     7: 'aorta',       8: 'pancreas'}

    logging.info('Synapse Trainer initalied !')

  def _save_best(self, epoch, best_metric, labels):
     dice_best = list(best_metric[:,0])
     hd95_best = list(best_metric[:,1])
     params = {
       'epoch' : epoch,
       'state_dict' : self.model.state_dict(),
       'optimizer_state_dict' : self.optimizer.state_dict(),
       'dice_list' : dice_best,
       'hd95_list' : hd95_best,
       'labels'    : labels,
       'hyper': self.hyper
     }
     torch.save(params, self.save_epoch_path)
     logging.info(f'\t save best epoch success!')

  def test(self):
    logging.info('start testing...')
    train_epoch(model=self.model, criterion=self.criterion, optimizer=self.optimizer, scheduler=self.scheduler,
                hyper=self.hyper,test_mode=True)
    logging.info('test train epoch success !')
    labels = [label for label in self.labels.values()][:self.num_classes]
    m = inference(model=self.model, hyper=self.hyper,labels=labels[1:], save_infrence=False, test_mode=True)
    # print(m)
    # print(m.shape)
    logging.info('test inference success !')

  def train(self, continue_train=False, epoch=None):

    labels = [label for label in self.labels.values()][:self.num_classes]
    if epoch is None:
       epoch_num = self.epoch_num
    else:
       epoch_num = epoch

    epoch_start = 0
    best_metric = np.zeros((8,2),dtype=np.float32)

    if continue_train:
      if os.path.exists(self.save_epoch_path):
        params = torch.load(self.save_epoch_path)
        dice_list = params['dice_list']
        hd95_list = params['hd95_list']
        epoch_start = params['epoch']
        logging.info(f'continue training from best epoch {epoch_start}')
        for i, organ in enumerate(params['labels']):
          logging.info(f'\t {organ} : dice={dice_list[i]},hd95={hd95_list[i]}')
        self.model.load_state_dict(params['state_dict'])
        self.optimizer.load_state_dict(params['optimizer_state_dict'])
        del params
      else:
         params = torch.load(os.path.join(self.save_path,"epoch.pth"))
         epoch_start = params['epoch']
         logging.info(f'continue training from epoch {epoch_start}')
         self.model.load_state_dict(params['state_dict'])
         self.optimizer.load_state_dict(params['optimizer_state_dict'])
         del params
    
    logging.info("---------- start trainning ----------")
    for epoch in range(epoch_start, epoch_num):
        t_tick = time.time()
        logging.info(f'epoch {epoch}/{epoch_num}:')
        logging.info(f'\t train:')
        train_epoch(model=self.model, criterion=self.criterion, optimizer=self.optimizer,scheduler=self.scheduler,
                    hyper=self.hyper,test_mode=False)
        if (epoch + 1)%self.hyper['save_frequncy'] == 0:
          params = {
                    'epoch' : epoch,
                    'state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict()
                    }
          torch.save(params, os.path.join(self.save_path,"epoch.pth"))
          del params
        

        do_inference = (epoch >= epoch_num//2 and epoch < epoch_num - 6 and (epoch+1) % self.eval_frequncy == 0) or (epoch > epoch_num - 6 and epoch%2==0) or epoch == epoch_num-1
        # do_inference = (epoch == epoch_num - 1)
        if do_inference:
          logging.info('\t inference:')
          
          save_infrence = False
          if epoch_num > int(epoch_num*0.8):
             save_infrence = True

          metric = inference(model=self.model, hyper=self.hyper,labels=labels[1:],
                             save_infrence=save_infrence, test_mode=False)
          if np.mean(metric[:,0]) > np.mean(best_metric[:,0]) or np.mean(metric[:,1]) < np.mean(metric[:,1]):
            best_metric = metric
            self._save_best(epoch,best_metric,labels[1:])

        logging.info(f'\t time: {(time.time() - t_tick):.2f} s')

    result_data = np.asarray(best_metric)
    result = pd.DataFrame(result_data,index=labels[1:],columns=['dice','hd95'])
    date = time.strftime("%Y%m%d_%H%M",time.localtime(time.time()))
    result.to_csv(os.path.join(self.save_path,f"test_result_{date}.csv"))






