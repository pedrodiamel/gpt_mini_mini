from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING


@dataclass
class ProjectConfig:
    path: str = MISSING
    datatime: str = MISSING
    name_prefix: str = MISSING
    version: float = 0.01


@dataclass
class DataConfig:
    path: str = MISSING
    name_dataset: str = MISSING
    epochs: int = 10
    batch_size: int = 32
    workers: int = 3


@dataclass
class ModelCheckpointConf:
    print_freq: int = 100
    resume: str = MISSING
    snapshot: int = 5
    verbose: bool = True


@dataclass
class ModelConfig:
    arch: str = MISSING
    block_size: int = 8
    n_layers: int = 6
    n_embd: int = 32
    n_heads: int = 4
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pretrained: bool = False


@dataclass
class TrainConfig:
    lr: float = 1e-4
    momentun: float = 0.9
    cuda: bool = True
    gpu: int = 0
    parallel: bool = False
    loss: str = MISSING
    opt: str = MISSING
    grad_clip: float = 0.1
    scheduler: str = MISSING
    compiler: bool = False


@dataclass
class EvalConfig:
    eval: bool = False
    test: bool = False


@dataclass
class AugmentationConfig:
    transforms_train: Any = None
    transforms_val: Any = None
    transforms_test: Any = None


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: ModelCheckpointConf = field(default_factory=ModelCheckpointConf)
    seed: int = 123456  # Seed for generators
