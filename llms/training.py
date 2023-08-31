import datetime
import json
import os
import random

import numpy as np
import pandas as pd

# TORCH MODULE
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from . import log, utils

# LOCAL MODULES
from .configs.hydra_config import Config
from .datasets.datasets import CharDataset
from .transformer import NeuralNetTransformer

logger = log.get_logger(__name__)


def train(cfg: Config):
    logger.info("Training transformer model {}!!!".format(datetime.datetime.now()))
    random.seed(cfg.seed)

    project_path = os.path.expanduser(cfg.project.path)
    if os.path.isdir(project_path) is not True:
        os.makedirs(project_path)

    project_name = "{}_{}_v{}".format(
        cfg.project.name_prefix,
        cfg.model.arch,
        cfg.project.version,
    )

    # Check path project
    project_pathname = os.path.expanduser(os.path.join(project_path, project_name))
    if os.path.exists(project_pathname) is not True:
        os.makedirs(project_pathname)
    else:
        response = input("Do you want to remove all files in this folder? (y/n): ")
        if response.lower() == "y":
            utils.remove_files(os.path.join(project_pathname, "models"))

    train_data = CharDataset(
        pathname=os.path.join(cfg.data.path, cfg.data.name_dataset),
        block_size=cfg.model.block_size,
        train=True,
        download=True,
    )

    train_random_sampler = RandomSampler(train_data)

    # Load data
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        sampler=train_random_sampler,
        num_workers=cfg.data.workers,
        pin_memory=cfg.trainer.cuda,
        drop_last=True,
    )

    val_data = CharDataset(
        pathname=os.path.join(cfg.data.path, cfg.data.name_dataset),
        block_size=cfg.model.block_size,
        train=False,
        download=False,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.workers,
        pin_memory=cfg.trainer.cuda,
        drop_last=False,
    )

    logger.info("Load datset")
    logger.info("Train dataset size: {}".format(len(train_data)))
    logger.info("Val dataset size: {}".format(len(val_data)))

    # Create and load neural net training class
    network = NeuralNetTransformer(
        pathproject=project_path,
        nameproject=project_name,
        no_cuda=not cfg.trainer.cuda,
        parallel=cfg.trainer.parallel,
        seed=cfg.seed,
        print_freq=cfg.checkpoint.print_freq,
        gpu=cfg.trainer.gpu,
        compiler=cfg.trainer.compiler,
    )

    network.create(
        arch=cfg.model.arch,
        vocab_size=train_data.vocab_size,
        block_size=cfg.model.block_size,
        n_layers=cfg.model.n_layers,
        n_embd=cfg.model.n_embd,
        n_heads=cfg.model.n_heads,
        loss=cfg.trainer.loss,
        lr=cfg.trainer.lr,
        optimizer=cfg.trainer.opt,
        lrsch=cfg.trainer.scheduler,
        grad_clip=cfg.trainer.grad_clip,
        pretrained=cfg.model.pretrained,
    )

    # Set cuda cudnn benchmark true
    cudnn.benchmark = True

    # Resume model
    if cfg.checkpoint.resume:
        network.resume(os.path.join(network.pathmodels, cfg.checkpoint.resume))

    logger.info("Load model: ")
    logger.info(network)

    # training neural net
    network.fit(train_loader, val_loader, cfg.data.epochs, cfg.checkpoint.snapshot)

    logger.info("DONE!!!")
