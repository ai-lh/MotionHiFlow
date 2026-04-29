import torch
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

def load_model(path, ckpt_name='net_best_fid'):
    path = Path(path)
    
    # load config and instantiate model
    model_cfg = OmegaConf.load(path / 'config.yaml')
    model = instantiate(model_cfg)
    
    # load weights
    ckpt_path = path / 'checkpoints' / f'{ckpt_name}.tar'
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # get state_dict (compatibility handling)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # load parameters
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith('text_encoder.')]
    if missing or unexpected:
        warnings.warn(f"Key mismatch!\nMissing: {missing}\nUnexpected: {unexpected}")
    
    print(f"Model loaded from {ckpt_path}")
    
    model.eval() 
    
    return model