from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING


@dataclass
class ProjectConfig:
    path: str = MISSING
    name_prefix: str = MISSING
    version: float = 0.01


@dataclass
class DataConfig:
    path: str = MISSING
    name_dataset: str = MISSING
    epochs: int = 10
    batch_size: int = 32
    workers: int = 3
    auto_balance: bool = False


@dataclass
class ModelCheckpointConf:
    print_freq: int = 100
    pretrained: bool = False
    pretrained_path: str = MISSING
    snapshot: int = 5
    verbose: bool = True


@dataclass
class ModelConfig:
    arch: str = MISSING
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    finetuning: bool = False


@dataclass
class TrainConfig:
    lr: float = 1e-4
    momentun: float = 0.9
    cuda: bool = True
    gpu: int = 0
    parallel: bool = False
    loss: str = MISSING
    opt: str = MISSING
    scheduler: str = MISSING


@dataclass
class AugmentationConfig:
    transforms_train: Any = None
    transforms_val: Any = None
    transforms_test: Any = None


@dataclass
class Config:
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainConfig = TrainConfig()
    checkpoint: ModelCheckpointConf = ModelCheckpointConf()
    seed: int = 123456  # Seed for generators
