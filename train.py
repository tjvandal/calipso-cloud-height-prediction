import sys
import os
import time
import argparse

from torch.utils.tensorboard import SummaryWriter # there is a bug with summarywriter importing
from collections import OrderedDict

import numpy as np
import torch
torch.cuda.empty_cache()

import torch.nn as nn
from torch import optim
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import dataloader
from nexai.train.base import CloudHeightTrainer

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning' 


def setup(rank, world_size, port):
    '''
    Setup multi-gpu processing group
    Args:
        rank: current rank
        world_size: number of processes
        port: which port to connect to
    Returns:
        None
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
    
def get_model():
    model = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, 1)
    )
    return model
'''
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
    )
    return model
'''

def train_net(params, rank=0):
    # set device
    #if not device:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = rank #% N_DEVICES
    print(f'in train_net rank {rank} on device {device}')

    if params['ngpus'] > 1:
        distribute = True
    else:
        distribute = False

    dataset_train = dataloader.CalipsoGOES(params['data_path'])
    dataset_valid = dataset_train
    
    data_params = {'batch_size': params['batch_size'] // params['ngpus'], 'shuffle': True,
                   'num_workers': 8, 'pin_memory': True}
    training_generator = data.DataLoader(dataset_train, **data_params)
    val_generator = data.DataLoader(dataset_valid, **data_params)

    model = get_model() 
    
    if distribute:
        model = DDP(model.to(device), device_ids=[device], find_unused_parameters=True)
    
    trainer = CloudHeightTrainer(model, 
                                 params['model_path'], 
                                 lr=params['lr'])

    trainer.model.to(device)
    trainer.load_checkpoint()

    print(f'Start Training')

    while trainer.global_step < params['max_iterations']:
        running_loss = 0.0
        t0 = time.time()
        for batch_idx, (images, heights) in enumerate(training_generator):
            images = images.to(device)
            heights = heights.to(device)

            log = False
            if (trainer.global_step % params['log_step'] == 0):
                log=True

            train_loss = trainer.step(images, heights, 
                                      log=log, train=True)
    
            if np.isinf(train_loss.cpu().detach().numpy()):
                print(train_loss, I0.cpu().detach().numpy(), I1.cpu().detach().numpy().flatten())
                return

            if log:
                images, heights = next(iter(val_generator))
                valid_loss = trainer.step(images.to(device), heights.to(device), 
                                          log=log, train=False)
                print(f'Rank {trainer.rank} @ Step: {trainer.global_step-1}, Training Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}')

            if (trainer.global_step % params['checkpoint_step'] == 0) and (rank == 0):
                trainer.save_checkpoint()


def manual_experiment(args):
    train_net(vars(args))

def train_net_mp(rank, world_size, port, params):
    '''
    Setup and train on node
    '''
    setup(rank, world_size, port)
    train_net(params, rank=rank)

def run_training(args, world_size, port):
    #params['batch_size'] = params['batch_size'] // world_size
    if world_size > 1:
        mp.spawn(train_net_mp,
                 args=(world_size, port, vars(args)),
                 nprocs=world_size,
                 join=True)
        cleanup()
    else:
        train_net(vars(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/testing-4/", type=str)
    parser.add_argument("--data_path", default="data/calipso_goes_pairs_w_qa/", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument('--max_iterations',type=int, default=2000000, 
                        help='Number of training iterations')
    parser.add_argument("--log_step", default=1000, type=int)
    parser.add_argument("--checkpoint_step", default=1000, type=int)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--port", default=9009, type=int)

    args = parser.parse_args()
    if torch.cuda.device_count() < args.ngpus:
        print(f"Cannot running training because {args.ngpus} are not available.")

    run_training(args, args.ngpus, args.port)
