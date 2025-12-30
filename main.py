"""
E2E-IST (End-to-End Image Semantic Transmission) for Progressive Image Transmission

Main entry point
"""

from data.datasets import get_loader
from config import setup_argument_parser, Config
from model_utils import setup_model, setup_optimizer_and_scheduler
from train import train_model
from test import run_test_mode
from utils import seed_torch, logger_configuration


def main():
    """Main execution function"""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Determine SSF usage and create config
    config = Config(args)
    
    # Initialize
    seed_torch(config.seed)
    logger = logger_configuration(config, save_log=True)
    logger.info("Initializing E2E-IST...")
    
    # Setup model
    net = setup_model(args, config, logger)
    
    # Create data loaders
    train_loader, val_loader = get_loader(args, config)
    
    # Training or test mode
    if args.training:
        optimizer, scheduler = setup_optimizer_and_scheduler(net, config)
        train_model(args, net, optimizer, scheduler, train_loader, val_loader,
                   config, logger)
    else:
        run_test_mode(args, net, val_loader, config, logger)


if __name__ == '__main__':
    main()