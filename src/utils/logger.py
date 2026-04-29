"""
Universal Experiment Logger Module
==================================

A robust, unified logging interface designed for Deep Learning research. 
This module allows you to write logs once and dispatch them simultaneously to multiple 
backends (Weights & Biases, TensorBoard, SwanLab, Console, and Local Disk).

Key Features:
-------------
1. **Multi-Backend Support**: Seamless integration with popular tracking tools.
2. **TQDM Integration**: Updates console logs without breaking progress bars.
   - Metrics are automatically shown in the tqdm postfix.
   - Info messages are printed cleanly above the progress bar.
3. **Robust Image Logging**: The FileHandler automatically converts PyTorch Tensors 
   (GPU/CPU), Numpy arrays, or PIL images into PNG files and saves them locally.
4. **Unified API**: Simple interface for scalars, images, histograms, and configs.

Supported Backends:
-------------------
- `console`: TQDM-safe console output.
- `filehandler`: Saves text logs and images to local disk.
- `wandb`: Weights & Biases integration.
- `tensorboard`: TensorBoard integration.
- `swanlab`: SwanLab integration.

Directory Structure (FileHandler):
----------------------------------
logs/
└── <run_name>/
    ├── execution.log         # Text logs (metrics, info)
    └── images/               # Saved images
        ├── step_00010_val_sample.png
        └── step_00020_val_sample.png

Usage Example:
--------------
    >>> from tqdm.auto import tqdm
    >>> from src.utils.logger import UniversalLogger
    
    # 1. Initialize
    logger = UniversalLogger(
        project_name="my-project", 
        run_name="exp-001",
        backends=['console', 'filehandler', 'wandb']
    )
    
    # 2. TQDM Integration (Crucial Step)
    # Create the progress bar
    pbar = tqdm(range(100), desc="Training")
    # Link pbar to logger. This enables postfix updates and safe printing.
    logger.set_progress_bar(pbar)
    
    # 3. Training Loop
    for step in pbar:
        # Log Metrics: This updates the tqdm postfix (e.g., loss=0.5) automatically
        # and logs to W&B/File simultaneously.
        logger.log_dict({"train/loss": 0.5, "train/acc": 0.8}, step=step)
        
        if step % 10 == 0:
            # Log Info: Prints SAFELY above the progress bar
            logger.info(f"Checkpoint saved at step {step}")
            
            # Log Images: Automatically handles Tensor/Numpy conversion
            img_tensor = torch.rand(3, 256, 256) 
            logger.log_image("val/output", img_tensor, step=step)
            
    # 4. Finish
    logger.close()
"""

import os
import json
import time
import logging
import shutil
import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# --- Optional Third-party libraries ---
try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import swanlab
except ImportError:
    swanlab = None


# --- 1. Abstract Base Handler ---

class BaseHandler(ABC):
    """
    Abstract base class for all logging handlers.
    Defines the interface that UniversalLogger uses to interact with different logging backends.
    """

    @abstractmethod
    def __init__(self, project_name: str, run_name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Logs a dictionary of metrics (scalars)."""
        pass

    @abstractmethod
    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        """Logs an image."""
        pass

    @abstractmethod
    def log_histogram(self, tag: str, values: Any, step: int):
        """Logs a histogram of values."""
        pass
    
    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        """Logs the experiment configuration."""
        pass

    @abstractmethod
    def info(self, message: str):
        """Logs an info message."""
        pass

    @abstractmethod
    def close(self):
        """Closes the handler and finalizes logging."""
        pass


# --- 2. Concrete Handler Implementations ---

class FileHandler(BaseHandler):
    """
    Handler that writes logs to a specified file and saves images to disk.
    
    Directory Structure:
        logs/
        └── run_name/
            ├── execution.log
            └── images/
                └── step_001_tag_name.png
    """
    def __init__(self, run_name: str, **kwargs):
        # 1. Setup Log Directory
        self.root_dir = Path(kwargs.get('log_dir', 'logs')) / run_name
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Setup Image Directory
        self.image_dir = self.root_dir / "files"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # 3. Setup File Logger
        filename = kwargs.get('filename')
        if not filename:
            self.log_file = self.root_dir / "execution.log"
        else:
            self.log_file = Path(filename)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a dedicated logger to avoid global scope pollution
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # Prevent propagation to root logger

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(str(self.log_file), mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"--- [FileHandler] Logs: {self.log_file} | Images: {self.image_dir} ---")

    def _convert_to_pil(self, image: Any) -> Image.Image:
        """Helper: Converts various input formats (Tensor, Numpy) to a standard PIL Image."""
        # 1. Handle PyTorch Tensors
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.dim() == 4: # Batch: taking first image (B, C, H, W) -> (C, H, W)
                image = image[0]
            if image.dim() == 3 and image.shape[0] in [1, 3, 4]: # CHW -> HWC
                image = image.permute(1, 2, 0)
            image = image.numpy()

        # 2. Handle Numpy Arrays
        if isinstance(image, np.ndarray):
            # If float and range [0, 1], scale to [0, 255]
            if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Handle CHW vs HWC heuristic for Numpy
            if image.ndim == 3 and image.shape[0] in [1, 3, 4] and image.shape[2] > 4:
                image = np.transpose(image, (1, 2, 0))
            
            image = image.astype(np.uint8)
            
            # Handle Grayscale (H, W, 1) -> (H, W)
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze(2)

            return Image.fromarray(image)

        # 3. Handle PIL Images
        if isinstance(image, Image.Image):
            return image

        raise ValueError(f"Unsupported image type: {type(image)}")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        msg = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"[Step {step}] {msg}")

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        try:
            pil_img = self._convert_to_pil(image)
            
            # Sanitize tag for filename (replace / with _)
            safe_tag = tag.replace("/", "_").replace("\\", "_").replace(" ", "_")
            filename = f"step_{step:05d}_{safe_tag}.png"
            save_path = self.image_dir / filename
            
            pil_img.save(save_path)
            
            log_msg = f"[Step {step}] Image saved: '{filename}'"
            if caption:
                log_msg += f" (Caption: {caption})"
            self.logger.info(log_msg)
            
        except Exception as e:
            self.logger.error(f"[Step {step}] Failed to save image '{tag}': {e}")

    def log_histogram(self, tag: str, values: Any, step: int):
        # File logging is not suitable for raw histogram data, just logging stats
        if isinstance(values, (torch.Tensor, np.ndarray)):
            vals = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
            stats = f"mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}"
            self.logger.info(f"[Step {step}] Histogram '{tag}': {stats}")

    def info(self, message: str):
        self.logger.info(message)

    def log_config(self, config: Dict[str, Any]):
        config_str = json.dumps(config, indent=2, sort_keys=True, default=str)
        self.logger.info(f"Configuration Logged:\n{config_str}")

    def close(self):
        self.logger.info("--- [FileHandler] closed. ---")
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


class PythonLoggingHandler(BaseHandler):
    """
    Adapter handler that forwards log messages to Python's root/standard logging.
    Useful when integrating with frameworks like Hydra.
    """
    def __init__(self, run_name: str, **kwargs):
        self.logger = logging.getLogger(run_name)
        self.logger.info(f"--- [PythonLoggingHandler] activated. ---")
    
    def info(self, message: str):
        self.logger.info(message)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        msg = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"[Step {step}] {msg}")

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        self.logger.info(f"[Step {step}] Image logged: '{tag}'")

    def log_histogram(self, tag: str, values: Any, step: int):
        self.logger.info(f"[Step {step}] Histogram logged: '{tag}'")
    
    def log_config(self, config: Dict[str, Any]):
        self.logger.info(f"Config: {config}")

    def close(self):
        pass


class TqdmConsoleHandler(BaseHandler):
    """
    Integrates with `tqdm` to update progress bars without breaking the console layout.
    """
    def __init__(self, project_name: str, run_name: str, **kwargs):
        self.run_name = run_name
        self._pbar: Optional[tqdm] = None
        self._postfix_metrics: Dict[str, Any] = {}
        tqdm.write(f"--- [TqdmConsoleHandler] Project: {project_name}, Run: {run_name} ---")

    def set_progress_bar(self, pbar: tqdm):
        """Sets the tqdm instance to be controlled by this handler."""
        pbar.set_description(f"\033[32m({self.run_name})\033[0m | {pbar.desc}")
        self._pbar = pbar

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        formatted_metrics = {}
        for k, v in metrics.items():
            # Shorten keys (e.g., train/loss -> loss)
            simple_key = k.split('/')[-1]
            if isinstance(v, (float, np.floating)):
                formatted_metrics[simple_key] = f"{v:.4f}"
            else:
                formatted_metrics[simple_key] = v
        
        if self._pbar:
            self._postfix_metrics.update(formatted_metrics)
            self._pbar.set_postfix(self._postfix_metrics, refresh=True)
        else:
            msg = ", ".join([f"{k}: {v}" for k, v in formatted_metrics.items()])
            tqdm.write(f"[Step {step}] {msg}")
            
    def info(self, message: str):
        tqdm.write(f"[INFO] {message}")

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        tqdm.write(f"[Step {step}] Image logged: '{tag}'")

    def log_histogram(self, tag: str, values: Any, step: int):
        tqdm.write(f"[Step {step}] Histogram logged: '{tag}'")

    def log_config(self, config: Dict[str, Any]):
        config_str = json.dumps(config, indent=2, sort_keys=True, default=str)
        tqdm.write(f"Configuration:\n{config_str}")

    def close(self):
        if self._pbar:
            try:
                self._pbar.close()
            except Exception:
                pass
        tqdm.write("--- [TqdmConsoleHandler] closed.")


class WandbHandler(BaseHandler):
    """Handler for Weights & Biases (wandb)."""
    def __init__(self, project_name: str, run_name: str, config: Optional[Dict[str, Any]] = None, save_code: bool = True, **kwargs):
        if not wandb:
            raise ImportError("wandb is not installed. Please run 'pip install wandb'.")
        
        # Filter kwargs for wandb.init
        wandb_init_params = inspect.signature(wandb.init).parameters.keys()
        wandb_kwargs = {k: v for k, v in kwargs.items() if k in wandb_init_params}

        log_dir = Path(kwargs.get('log_dir', './logs')) / run_name / 'wandb'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            settings = wandb.Settings(code_dir="." if save_code else None)
            self.run = wandb.init(
                project=project_name, 
                name=run_name, 
                config=config, 
                reinit=True, 
                dir=str(log_dir),
                settings=settings,
                **wandb_kwargs
            )
            
            if save_code and self.run:
                self._backup_code(log_dir)

        except Exception as e:
            print(f"[WandbHandler] Failed to initialize: {e}")
            self.run = None

    def _backup_code(self, log_dir: Path):
        """Helper to save code to WandB and local backup."""
        try:
            workspace_root = Path(os.getcwd()) # Use CWD usually
            
            # Save to WandB
            if self.run:
                self.run.log_code(root=str(workspace_root), include_fn=lambda path: path.endswith(('.py', '.yaml', '.yml', '.json')))
            
        except Exception as e:
            print(f"[WandbHandler] Warning: Code backup failed: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self.run:
            self.run.log(metrics, step=step)

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        if self.run:
            self.run.log({tag: wandb.Image(image, caption=caption)}, step=step)
        
    def log_histogram(self, tag: str, values: Any, step: int):
        if self.run:
            self.run.log({tag: wandb.Histogram(values)}, step=step)
            
    def info(self, message: str):
        pass

    def log_config(self, config: Dict[str, Any]):
        if self.run:
            self.run.config.update(config, allow_val_change=True)

    def close(self):
        if self.run:
            self.run.finish()


class TensorBoardHandler(BaseHandler):
    """Handler for TensorBoard."""
    def __init__(self, project_name: str, run_name: str, **kwargs):
        if not SummaryWriter:
            raise ImportError("TensorBoard is not installed. Please run 'pip install tensorboard'.")
        
        tb_init_params = inspect.signature(SummaryWriter).parameters.keys()
        tb_kwargs = {k: v for k, v in kwargs.items() if k in tb_init_params}
        
        log_dir = Path(kwargs.get('log_dir', './logs')) / run_name / 'tensorboard'
        
        try:
            self.writer = SummaryWriter(log_dir=str(log_dir), **tb_kwargs)
            print(f"[TensorBoardHandler] Logs at: {log_dir}")
        except Exception as e:
            print(f"[TensorBoardHandler] Initialization failed: {e}")
            self.writer = None
            
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if not self.writer: return
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number, torch.Tensor)):
                 self.writer.add_scalar(k, v, step)

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        if self.writer:
            # Handle formats: TB expects CHW for tensors
            # We use a simplified heuristic here or rely on add_image's auto-detect if simpler
            if isinstance(image, Image.Image): 
                image = np.array(image)
            
            dataformats = 'CHW'
            if isinstance(image, np.ndarray):
                # Heuristic: If last dim is 1, 3, or 4, assume HWC
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                      dataformats = 'HWC'
            
            self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_histogram(self, tag: str, values: Any, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def info(self, message: str):
        pass

    def log_config(self, config: Dict[str, Any]):
        if self.writer:
            config_str = json.dumps(config, indent=2, default=str)
            self.writer.add_text('config', f"```json\n{config_str}\n```", 0)

    def close(self):
        if self.writer:
            self.writer.close()


class SwanLabHandler(BaseHandler):
    """Handler for SwanLab."""
    def __init__(self, project_name: str, run_name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        if not swanlab:
            raise ImportError("swanlab is not installed. Please run 'pip install swanlab'.")
        
        swan_init_params = inspect.signature(swanlab.init).parameters.keys()
        swan_kwargs = {k: v for k, v in kwargs.items() if k in swan_init_params}
        
        log_dir = Path(kwargs.get('log_dir', './logs')) / 'swanlab' / run_name

        try:
            self.run = swanlab.init(
                project=project_name,
                experiment_name=run_name,
                config=config,
                logdir=str(log_dir),
                **swan_kwargs
            )
        except Exception as e:
            print(f"[SwanLabHandler] Initialization failed: {e}")
            self.run = None

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self.run:
            self.run.log(metrics, step=step)

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        if self.run:
            self.run.log({tag: swanlab.Image(image, caption=caption)}, step=step)

    def log_histogram(self, tag: str, values: Any, step: int):
        # SwanLab may not have a direct histogram primitive identical to TB, logging stats is safer
        if self.run and isinstance(values, (np.ndarray, torch.Tensor)):
            vals = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
            self.run.log({
                f'{tag}/mean': float(vals.mean()), 
                f'{tag}/std': float(vals.std()),
                f'{tag}/max': float(vals.max()), 
                f'{tag}/min': float(vals.min())
            }, step=step)
    
    def info(self, message: str):
        pass

    def log_config(self, config: Dict[str, Any]):
        if self.run:
            self.run.config.update(config)

    def close(self):
        if self.run and hasattr(self.run, 'finish'):
            self.run.finish()


# --- 3. The Main UniversalLogger Class ---

class UniversalLogger:
    """
    A universal, extensible logger that delegates logging tasks to multiple handlers.
    """
    _handler_class_map = {
        'python_logging': PythonLoggingHandler,
        'console': TqdmConsoleHandler,
        'wandb': WandbHandler,
        'tensorboard': TensorBoardHandler,
        'swanlab': SwanLabHandler,
        'filehandler': FileHandler, 
    }

    def __init__(self, project_name: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, 
                 backends: List[str] = ['console', 'wandb'], **kwargs):
        
        if run_name is None:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        
        self.handlers: List[BaseHandler] = []
        self.tqdm_handler: Optional[TqdmConsoleHandler] = None
        
        # Initialize Handlers
        for backend_name in backends:
            handler_class = self._handler_class_map.get(backend_name.lower())
            if handler_class:
                try:
                    handler_instance = handler_class(project_name=project_name, run_name=run_name, config=config, **kwargs)
                    self.handlers.append(handler_instance)
                    
                    if isinstance(handler_instance, TqdmConsoleHandler):
                        self.tqdm_handler = handler_instance
                except Exception as e:
                    print(f"Error loading handler '{backend_name}': {e}")
            else:
                print(f"Warning: Unknown backend '{backend_name}' ignored.")
        
        if config:
            self.log_config(config)

    def set_progress_bar(self, pbar: tqdm):
        """Register an external tqdm progress bar instance."""
        if self.tqdm_handler:
            self.tqdm_handler.set_progress_bar(pbar)

    def info(self, message: str):
        """Log generic information (e.g., status updates)."""
        for h in self.handlers:
            h.info(message)

    def log_dict(self, data: Dict[str, Any], step: int, tqdm_keys: Optional[Union[List[str], str]] = None):
        """
        Log a dictionary of metrics.
        
        Args:
            data: Dictionary of metric name -> value.
            step: Current step.
            tqdm_keys: Control which metrics are shown in the console progress bar.
                       - None (Default): Show ALL metrics in data.
                       - List[str]: Only show keys present in this list (e.g. ['loss']).
                       - []: Show NO metrics in tqdm (keep logging to wandb/file).
        """
        # 1. Clean data (handle single-element tensors)
        clean_metrics = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                clean_metrics[k] = v.item() if v.numel() == 1 else v
            else:
                clean_metrics[k] = v
        
        # 2. Handle tqdm_keys normalization
        target_keys = None
        if tqdm_keys is not None:
            if isinstance(tqdm_keys, str):
                target_keys = {tqdm_keys}
            else:
                target_keys = set(tqdm_keys)

        # 3. Dispatch to handlers
        for h in self.handlers:
            # Special filtering logic for TqdmConsoleHandler
            if isinstance(h, TqdmConsoleHandler):
                if target_keys is not None:
                    # Filter: Only pass keys that exist in both clean_metrics and target_keys
                    filtered_data = {k: v for k, v in clean_metrics.items() if k in target_keys}
                    if filtered_data:
                        h.log_metrics(filtered_data, step=step)
                    # If target_keys is [], we simply don't call log_metrics on tqdm handler
                else:
                    # Default: Log everything
                    h.log_metrics(clean_metrics, step=step)
            else:
                # All other handlers (WandB, File, TB) get the FULL dataset
                h.log_metrics(clean_metrics, step=step)

    def log_scalar(self, tag: str, value: Any, step: int, show_in_tqdm: bool = True):
        """
        Syntactic sugar for logging a single value.
        Args:
            show_in_tqdm: If False, this scalar will go to WandB/File but NOT the console bar.
        """
        tqdm_keys = [tag] if show_in_tqdm else []
        self.log_dict({tag: value}, step=step, tqdm_keys=tqdm_keys)

    def log_image(self, tag: str, image: Any, step: int, caption: str = ''):
        for h in self.handlers:
            h.log_image(tag, image, step, caption)
        
    def log_histogram(self, tag: str, values: Any, step: int):
        for h in self.handlers:
            h.log_histogram(tag, values, step)
        
    def log_config(self, config: Dict[str, Any]):
        for h in self.handlers:
            h.log_config(config)

    def close(self):
        print("\n[UniversalLogger] Closing all loggers...")
        for h in self.handlers:
            h.close()
        print("[UniversalLogger] Done.")


# --- 4. Helper for Third-Party Library Integration ---

class TqdmLoggingRedirectHandler(logging.Handler):
    """
    A standard logging.Handler that redirects logs to tqdm.write.
    Use this to make other libraries (like Transformers/PyTorch) play nice with your progress bar.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# --- Example Usage ---

if __name__ == '__main__':
    # Configuration
    config = {
        "learning_rate": 0.001,
        "epochs": 1,
        "batch_size": 32,
        "architecture": "ResNet18"
    }

    # Initialize Logger
    logger = UniversalLogger(
        project_name="polishing-demo",
        run_name=f"test_run_{time.strftime('%y%m%d-%H%M')}",
        config=config,
        backends=['console', 'filehandler', 'tensorboard'], # Add 'wandb' to test if installed
        log_dir='./logs'
    )

    logger.info("Starting training loop simulation...")

    global_step = 0
    total_batches = 10

    # Simulate Training
    pbar = tqdm(range(total_batches), desc="Training")
    logger.set_progress_bar(pbar)

    for batch_idx in pbar:
        time.sleep(0.05) 
        global_step += 1
        
        # Log metrics
        loss = 1.0 / (global_step + 1) + np.random.random() * 0.1
        logger.log_dict({"train/loss": loss}, step=global_step)
        
        # Simulate Image Logging (last step)
        if batch_idx == total_batches - 1:
            # Create a fake tensor image (3, 64, 64) float [0, 1]
            fake_img_tensor = torch.rand(3, 64, 64) 
            logger.log_image("val/generated_sample", fake_img_tensor, step=global_step, caption="Random Noise Tensor")

    logger.info("Training finished.")
    logger.close()