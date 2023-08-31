# Exemplo
# python cli/train.py +configs=gptmm_v1

import hydra

from hydra.core.config_store import ConfigStore
from llms import log
from llms.configs.hydra_config import Config
from llms.training import train
from omegaconf import DictConfig, OmegaConf

logger = log.get_logger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: Config) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
