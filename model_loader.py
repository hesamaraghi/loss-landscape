import os
import torch
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    else:
        net = torch.load(model_file)
    return net


#TODO #45 
# models = model_factory.factory(cfg) 
# runners = [Runner.load_from_checkpoint(artifact_dir, cfg=cfg, model=model) for artifact_dir, cfg, model in zip(artifact_dirs, cfgs, models)]
