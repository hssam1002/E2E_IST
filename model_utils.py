"""
Model utility functions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from net.network import E2E_SwinJSCC
from config import DEFAULT_RHO_INIT


def get_alpha_sequence(L, mode='linear', device='cuda'):
    """
    Generate alpha sequence (normalized to sum to 1).
    
    Args:
        L (int): Sequence length
        mode (str): Generation mode ('base', 'linear', 'inverse', 'square', 'exponential', 'uniform')
        device (str): Device for tensor
    
    Returns:
        torch.Tensor: Normalized alpha sequence
    """
    if mode == 'base':
        return torch.zeros(L, dtype=torch.float32, device=device)
    elif mode == 'linear':
        v = torch.linspace(L, 1, steps=L)
    elif mode == 'inverse':
        v = 1.0 / torch.arange(1, L + 1, dtype=torch.float32)
    elif mode == 'square':
        v = 1.0 / (torch.arange(1, L + 1, dtype=torch.float32) ** 2)
    elif mode == 'exponential':
        v = torch.exp(-torch.arange(0, L, dtype=torch.float32))
    elif mode == 'uniform':
        v = torch.ones(L, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown alpha mode: {mode}")
    
    return (v / v.sum()).to(device)


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


def find_model_path(model_dir, progressive_mode, alpha_mode=None):
    """
    Find model file path for given progressive_mode.
    
    Args:
        model_dir (str): Directory containing models
        progressive_mode (str): Progressive mode name
        alpha_mode (str, optional): Alpha mode name (only for alm, adaptive-alm)
    
    Returns:
        str or None: Model file path, or None if not found
    """
    if not model_dir or not os.path.exists(model_dir):
        return None
    
    # Alpha mode needed for alm, adaptive-alm
    needs_alpha_mode = progressive_mode in ['alm', 'adaptive-alm']
    
    # Try multiple naming conventions
    candidates = []
    
    # 1. {progressive_mode}_{alpha_mode}.pth (for alm, adaptive-alm)
    if needs_alpha_mode and alpha_mode:
        candidates.append(os.path.join(model_dir, f"{progressive_mode}_{alpha_mode}.pth"))
    
    # 2. best_model_{alpha_mode}.pth (common naming convention)
    if alpha_mode:
        candidates.append(os.path.join(model_dir, f"best_model_{alpha_mode}.pth"))
    
    # 3. {progressive_mode}.pth
    candidates.append(os.path.join(model_dir, f"{progressive_mode}.pth"))
    
    # 4. best_model_base.pth (fallback for base modes)
    candidates.append(os.path.join(model_dir, "best_model_base.pth"))
    
    # Check candidates in order
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    
    return None


def freeze_parameters_for_adaptive(net, target='both'):
    """
    Freeze all parameters except SSF for adaptive mode.
    
    Args:
        net (nn.Module): Model
        target (str): SSF activation location ('enc', 'dec', 'both')
    """
    # Freeze all parameters
    for param in net.parameters():
        param.requires_grad = False
    
    # Enable SSF parameters only
    trainable_count = 0
    for name, param in net.named_parameters():
        is_ssf = 'ssf' in name
        
        is_target = False
        if target == 'both':
            is_target = True
        elif target == 'enc' and 'encoder' in name:
            is_target = True
        elif target == 'dec' and 'decoder' in name:
            is_target = True
        
        if is_ssf and is_target:
            param.requires_grad = True
            trainable_count += 1
    
    print(
        f"[*] Adaptive Mode: Base model frozen. "
        f"{trainable_count} SSF parameters are trainable."
    )


def get_use_ssf(progressive_mode):
    """
    Determine SSF usage based on progressive mode.
    
    Args:
        progressive_mode (str): Progressive mode value
    
    Returns:
        bool: Whether to use SSF (True for adaptive-* modes)
    """
    return progressive_mode.startswith('adaptive-')


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
    
    if args.progressive_mode.startswith('adaptive-'):
        freeze_parameters_for_adaptive(net, target=args.ssf_target)
    
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


def initialize_alm_parameters(args, config):
    """
    Initialize ALM parameters.
    
    Args:
        args: Parsed arguments
        config: Config object
    
    Returns:
        tuple: (lambda_l, rho, prev_h_norm, curr_h_norm)
    """
    C_total = config.encoder_kwargs['embed_dims'][-1]
    F = args.packet_size
    L_steps = (C_total + F - 1) // F
    lambda_l = torch.zeros(L_steps, device=config.device)
    rho = DEFAULT_RHO_INIT
    prev_h_norm = float('inf')
    curr_h_norm = 0.0
    return lambda_l, rho, prev_h_norm, curr_h_norm