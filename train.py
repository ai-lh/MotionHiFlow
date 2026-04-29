import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
import numpy as np
import random
from pathlib import Path
from hydra.utils import instantiate, get_class
from hydra.core.hydra_config import HydraConfig
from collections import defaultdict
from src.utils import load_model, seed_everything
import pandas as pd

def count_parameters(model: torch.nn.Module, human_readable: bool = True) -> str:
    total_params = float(sum(p.numel() for name, p in model.named_parameters() if "text_encoder." not in name.lower()))
    if human_readable:
        for unit in ['','K','M','B','T']:
            if total_params < 1024:
                return f"{total_params:.2f}{unit}"
            total_params /= 1024
        return f"{total_params:.2f}P"
    else:
        return str(total_params)

@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # 1. init set seed
    seed_everything(cfg.seed)

    logger = instantiate(cfg.logger)
    logger.info("--- Configuration ---")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logger.info("---------------------")


    logger.info(f"Initializing model...")
    model = instantiate(cfg.model)
    logger.info(f"Model parameters: {count_parameters(model)}")
    with open(Path(cfg.folder.run) / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(model.config))
    logger.info(f"Model config: \n{OmegaConf.to_yaml(model.config)}")
    logger.info(f"Model config saved to {Path(cfg.folder.run) / 'config.yaml'}")


    logger.info("Loading dataset...")
    mean = np.load(cfg.data.eval_mean)
    std = np.load(cfg.data.eval_std)

    train_dataset = instantiate(cfg.data.train_dataset, mean=mean, std=std)
    val_dataset = instantiate(cfg.data.val_dataset, mean=mean, std=std)
    eval_val_dataset = instantiate(cfg.data.eval_dataset, mean=mean, std=std)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, drop_last=True,
        num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True
    )
    eval_val_loader = torch.utils.data.DataLoader(
        eval_val_dataset, batch_size=32, drop_last=True,
        num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True, collate_fn=instantiate(cfg.data.eval_collate_fn)
    )


    logger.info("Loading evaluation wrapper...")
    eval_wrapper = instantiate(cfg.evaluator)


    logger.info(f"\nStarting training...")
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)
    Trainer = instantiate(
        cfg.trainer, model=model, logger=logger,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
        cfg=cfg, _recursive_=False
    )
    Trainer.train(
        train_loader, val_loader, eval_val_loader, 
        eval_wrapper, eval_func=instantiate(cfg.data.eval_func)
    )

    logger.info(f"\nTraining complete. Starting evaluation on test set...")
    from eval import main as eval_main
    eval_main(cfg)


if __name__ == "__main__":
    main()