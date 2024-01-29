import os
import cifar10.model_loader
import sys
from omegaconf import OmegaConf
import torch
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)


import model_factory
import os


def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    else:
        cfg_file = os.path.join(os.path.dirname(model_file), "cfg.yaml")
        cfg = OmegaConf.load(cfg_file)
        net = model_factory.factory(cfg)
        net.load_state_dict(torch.load(model_file))
    return net


#TODO #45 
# models = model_factory.factory(cfg) 
# runners = [Runner.load_from_checkpoint(artifact_dir, cfg=cfg, model=model) for artifact_dir, cfg, model in zip(artifact_dirs, cfgs, models)]
