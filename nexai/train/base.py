import os, sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter # there is a bug with summarywriter importing

import torchvision

def scale_image(x):
    xmn = torch.min(x)
    xmx = torch.max(x)
    return (x - xmn) / (xmx - xmn)

class BaseTrainer(nn.Module):
    def __init__(self, model, 
                 model_path,
                 lr=1e-4,
                 device=None,
                 distribute=False,
                 rank=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.model_path = model_path
        self.lr = lr
        self.device = device
        self.distribute = distribute
        self.rank = rank
        
        # NEW
        self.scaler = torch.cuda.amp.GradScaler()
            
        self.checkpoint_filepath = os.path.join(model_path, 'checkpoint.pth.tar')
        if (rank == 0) and (not os.path.exists(model_path)):
            os.makedirs(model_path)
        
        self.global_step = 0
        self._set_optimizer()
        self._set_summary_writer()

        
    def _set_optimizer(self):
        # set optimizer
        #if self.rank == 0:
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-8)

    def _set_summary_writer(self):
        self.tfwriter_train = SummaryWriter(os.path.join(self.model_path, 'train', 'tfsummary'))
        self.tfwriter_valid = SummaryWriter(os.path.join(self.model_path, 'valid', 'tfsummary'))

    def load_checkpoint(self):
        filename = self.checkpoint_filepath
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            self.global_step = checkpoint['global_step']
            try:
                self.model.module.load_state_dict(checkpoint['model'])
            except:
                self.model.load_state_dict(checkpoint['model'])
                
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (Step {})"
                    .format(filename, self.global_step))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        
    def save_checkpoint(self):
        if self.distribute:
            state = {'global_step': self.global_step, 
                     'model': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        else:
            state = {'global_step': self.global_step, 
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.checkpoint_filepath)        

    def log_tensorboard(self):
        pass

    def get_tfwriter(self, train):
        if train:
            return self.tfwriter_train
        else:
            return self.tfwriter_valid
        
    def log_scalar(self, x, name, train=True):
        tfwriter = self.get_tfwriter(train)
        tfwriter.add_scalar(name, x, self.global_step)
    
    def log_image_grid(self, img, name, train=True, N=4):
        '''
        img of shape (N, C, H, W)
        '''
        tfwriter = self.get_tfwriter(train)
        img_grid = torchvision.utils.make_grid(img[:N])
        tfwriter.add_image(name, scale_image(img_grid), self.global_step)
        
        
class CloudHeightTrainer(BaseTrainer):
    def __init__(self,
                 model, 
                 model_path, 
                 lr=1e-4,
                 device=None,
                 distribute=False,
                 rank=0,
                 loss='L2',
                 fill_value=-9999.):
        BaseTrainer.__init__(self, model, model_path, lr=lr, 
                             device=device, distribute=distribute, rank=rank)

        # set loss functions
        self.loss = nn.MSELoss()
        self.bce = nn.BCELoss()

    def step(self, inputs, labels, log=False, train=True):
        '''
        Inputs shape: (N, C, H, W)
        Labels shape: (N, 1)
        '''
        output = self.model(inputs) 

        clouds = (labels != -9999.)
        class_loss = self.bce(output[:,:1], clouds.float())
        if sum(clouds) > 0:
            reg_loss = self.loss(labels[clouds], output[:,1:][clouds])
        else:
            reg_loss = 0
            
        total_loss = class_loss + reg_loss
 
        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        if log and (self.rank == 0):
            self.log_scalar(class_loss, "classification", train=train)
            self.log_scalar(reg_loss, "regression", train=train)
            self.log_scalar(total_loss, "loss", train=train)

        if train:
            self.global_step += 1

        return total_loss
