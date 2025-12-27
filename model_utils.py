"""
Model utility functions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
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


def find_model_path(model_dir, learning_mode):
    """
    Find model file path for given learning_mode.
    
    Args:
        model_dir (str): Directory containing models
        learning_mode (str): Learning mode name (rand_mask_1 or rand_mask_2)
    
    Returns:
        str or None: Model file path, or None if not found
    """
    if not model_dir or not os.path.exists(model_dir):
        return None
    
    # Try multiple naming conventions
    candidates = [
        os.path.join(model_dir, f"{learning_mode}.pth"),
        os.path.join(model_dir, f"best_model_{learning_mode}.pth"),
        os.path.join(model_dir, "best_model.pth")
    ]
    
    # Check candidates in order
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    
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
    # Get model parameters (handle DataParallel)
    if isinstance(net, nn.DataParallel):
        params = net.module.parameters()
    else:
        params = net.parameters()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, params),
        lr=config.lr
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[2000, 3500, 4500],  # 더 일찍 decay하여 더 많은 학습 기회 제공
        gamma=0.5  # 더 작은 decay factor로 점진적 감소
    )
    return optimizer, scheduler

