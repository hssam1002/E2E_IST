"""
Model utility functions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from net.network import E2E_SwinJSCC


def load_weights(net, path, strict=False):
    """
    Load pretrained weights.
    
    Args:
        net (nn.Module): Model to load weights into
        path (str): Path to weight file
        strict (bool): Strict mode (False recommended when SSF params are added)
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path not found: {path}")
    
    state_dict = torch.load(path)
    # Remove 'module.' prefix if saved with DataParallel
    new_state_dict = {
        k.replace('module.', ''): v 
        for k, v in state_dict.items()
    }
    
    missing, unexpected = net.load_state_dict(new_state_dict, strict=strict)
    print(f"[*] Weights loaded from {path}")
    print(f"    Missing Keys (Expected if Adaptive): {len(missing)}")
    print(f"    Unexpected Keys: {len(unexpected)}")


def find_model_path(model_dir, progressive_mode):
    """
    Find model file path for given progressive_mode.
    
    Args:
        model_dir (str): Directory containing models
        progressive_mode (str): Progressive mode name
    
    Returns:
        str or None: Model file path, or None if not found
    """
    if not model_dir or not os.path.exists(model_dir):
        return None
    
    candidate2 = os.path.join(model_dir, f"{progressive_mode}.pth")
    if os.path.exists(candidate2):
        return candidate2
    
    return None


def setup_model(args, config, logger):
    """
    Initialize and setup model.
    
    Args:
        args: Parsed arguments
        config: Config object
        logger: Logger object
    
    Returns:
        nn.Module: Configured model
    """
    net = E2E_SwinJSCC(args, config).to(config.device)
    
    if args.pretrained:
        load_weights(net, args.pretrained, strict=False)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    
    return net


def setup_optimizer_and_scheduler(net, config):
    """
    Setup optimizer and scheduler.
    
    Args:
        net (nn.Module): Model
        config: Config object
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10,
        factor=0.5,
        min_lr=1e-7,
        verbose=True
    )
    return optimizer, scheduler