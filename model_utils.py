"""
Model utility functions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
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
    
    state_dict = torch.load(path, map_location='cpu')
    # Remove 'module.' prefix if saved with DataParallel
    new_state_dict = {
        k.replace('module.', ''): v 
        for k, v in state_dict.items()
        if 'attn_mask' not in k  # Skip attn_mask buffers - they are auto-regenerated based on resolution
    }
    
    missing, unexpected = net.load_state_dict(new_state_dict, strict=strict)
    print(f"[*] Weights loaded from {path}")
    print(f"    Missing Keys (Expected if Adaptive): {len(missing)}")
    print(f"    Unexpected Keys: {len(unexpected)}")
    
    # After loading, update resolution to regenerate attn_mask buffers
    # This is handled automatically by the model's forward pass or update_resolution calls


def find_model_path(model_dir, progressive_mode):
    """
    Find model file path for given progressive_mode.
    
    Args:
        model_dir (str): Directory containing models
        progressive_mode (str): Progressive mode name (kept for compatibility, but best_model.pth is preferred)
    
    Returns:
        str or None: Model file path, or None if not found
    """
    if not model_dir or not os.path.exists(model_dir):
        return None
    
    # Look for best_model.pth first (preferred)
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    
    # Fallback: look for progressive_mode.pth
    mode_model_path = os.path.join(model_dir, f"{progressive_mode}.pth")
    if os.path.exists(mode_model_path):
        return mode_model_path
    
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


def setup_optimizer_and_scheduler(net, config, args):
    """
    Setup optimizer and scheduler.
    
    Args:
        net (nn.Module): Model
        config: Config object
        args: Parsed arguments
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Get model parameters (handle DataParallel)
    if isinstance(net, nn.DataParallel):
        params = net.module.parameters()
    else:
        params = net.parameters()
    
    # Setup optimizer
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, params),
            lr=config.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, params),
            lr=config.learning_rate,
            weight_decay=args.weight_decay if args.scheduler != 'ReduceLROnPlateau' else 0.0
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Setup scheduler
    if args.scheduler == 'Cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.tot_epoch,
            eta_min=config.learning_rate * 0.01
        )
    elif args.scheduler == 'MultiStep':
        milestones = [int(x.strip()) for x in args.scheduler_milestones.split(',')]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.scheduler_gamma
        )
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # PSNR is higher is better
            patience=10,
            factor=0.5,
            min_lr=config.learning_rate * 0.01,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    return optimizer, scheduler