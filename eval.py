import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
import numpy as np
import random
from os.path import join as pjoin
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


@hydra.main(version_base=None, config_path="./configs", config_name="test")
def main(cfg: DictConfig) -> None:
    # 1. init set seed
    seed_everything(cfg.seed)

    model = load_model(cfg.folder.run, cfg.ckpt_name)
    logging.info(f"Model loaded from checkpoint {cfg.ckpt_name} of {cfg.name}.")
    logging.info(f"Model parameters: {count_parameters(model)}")

    logging.info("Loading evaluation dataset...")
    mean = np.load(pjoin(cfg.data.eval_mean))
    std = np.load(pjoin(cfg.data.eval_std))

    logging.info("Loading evaluation wrapper...")
    eval_wrapper = instantiate(cfg.evaluator)
    replication_times = cfg.get('replication_times', 20) # get from configuration, default is 20

    logging.info("Loading evaluation dataset...")
    eval_test_dataset = instantiate(cfg.data.test_dataset, mean=mean, std=std)
    eval_test_loader = torch.utils.data.DataLoader(
        eval_test_dataset, batch_size=32, drop_last=True,
        num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True, collate_fn=instantiate(cfg.data.eval_collate_fn)
    )
    logging.info("Dataset loaded.")

    logging.info(f"\nStarting evaluation for {replication_times} replications...")
    Trainer = get_class(cfg.trainer._target_)
    all_metrics = Trainer.test(model, eval_test_loader, eval_wrapper, cfg, replication_times=replication_times, eval_func=instantiate(cfg.data.eval_func))


    logging.info("\n--- Evaluation Summary ---")
    # 5. Calculate and report final results
    summary_df = pd.DataFrame.from_dict(
        {k: [float(x) for x in v] for k, v in all_metrics.items()},
        orient='index'
    ).T
    # compute mean and 95% confidence interval (using replication_times)
    summary_df.loc['mean'] = summary_df.mean()
    summary_df.loc['conf'] = summary_df.apply(lambda col: np.std(col.values) * 1.96 / np.sqrt(replication_times))


    try:
        hydra_cfg = HydraConfig.get()
        add_name = f"_cs{cfg.cond_scale}_ts{cfg.time_steps}" if hydra_cfg.runtime.choices['model'] == 'tmdit' else ""
    except Exception:
        add_name = ""
        logging.info("Could not get HydraConfig, setting add_name to empty string.")
    save_path = pjoin(cfg.folder.run, 'eval', f'results_{cfg.ckpt_name}{add_name}_seed{cfg.seed}.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path)
    
    # log each metric as "name: mean ± conf" with 4 decimal places
    logging.info("Final metrics (mean ± 95% CI):")
    for metric in summary_df.columns:
        try:
            mean_val = float(summary_df.at['mean', metric]) # type: ignore
            conf_val = float(summary_df.at['conf', metric]) # type: ignore
            logging.info(f"{metric}: {mean_val:.4f} ± {conf_val:.4f}")
        except Exception:
            logging.info(f"{metric}: unable to format mean/conf")
    logging.info(f"Results saved to {save_path}")
    logging.info("---------------------")


if __name__ == "__main__":
    main()