"""
Logger utilities for DACER training
Provides TensorBoard and WandB logging support
"""

import os
import datetime
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class TensorBoardLogger:
    """TensorBoard logger for DACER training"""
    
    def __init__(self, log_dir: str, comment: str = ""):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory to save logs
            comment: Comment to append to log directory name
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is not installed. Install with: pip install tensorboard")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)
        
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value"""
        self.writer.add_scalar(key, value, step)
        
    def log_histogram(self, key: str, values, step: int) -> None:
        """Log a histogram"""
        self.writer.add_histogram(key, values, step)
        
    def log_params(self, params: Dict[str, Any], step: int = 0) -> None:
        """Log hyperparameters"""
        # Convert all params to strings for logging
        str_params = {k: str(v) for k, v in params.items()}
        self.writer.add_hparams(str_params, {})
        
    def close(self) -> None:
        """Close the TensorBoard writer"""
        if self.writer:
            self.writer.close()


class WandBLogger:
    """Weights & Biases logger for DACER training"""
    
    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None, 
                 name: Optional[str] = None, dir: Optional[str] = None):
        """
        Initialize WandB logger
        
        Args:
            project: WandB project name
            config: Configuration dictionary
            name: Run name
            dir: Directory to save logs
        """
        if not WANDB_AVAILABLE:
            raise ImportError("WandB is not installed. Install with: pip install wandb")
        
        # Initialize wandb run
        wandb.init(
            project=project,
            config=config or {},
            name=name,
            dir=dir
        )
        
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value"""
        wandb.log({key: value}, step=step)
        
    def log_histogram(self, key: str, values, step: int) -> None:
        """Log a histogram"""
        wandb.log({key: wandb.Histogram(values)}, step=step)
        
    def log_params(self, params: Dict[str, Any], step: int = 0) -> None:
        """Log hyperparameters (already done in init)"""
        # Params are already logged during init via config
        pass
        
    def close(self) -> None:
        """Close the WandB run"""
        wandb.finish()


class DummyLogger:
    """Dummy logger that does nothing - used when logging is disabled"""
    
    def __init__(self, *args, **kwargs):
        pass
        
    def log_scalar(self, key: str, value: float, step: int) -> None:
        pass
        
    def log_histogram(self, key: str, values, step: int) -> None:
        pass
        
    def log_params(self, params: Dict[str, Any], step: int = 0) -> None:
        pass
        
    def close(self) -> None:
        pass


def create_logger(logger_type: str, **kwargs) -> Union[TensorBoardLogger, WandBLogger, DummyLogger]:
    """
    Factory function to create a logger
    
    Args:
        logger_type: Type of logger ('tensorboard', 'wandb', 'none')
        **kwargs: Additional arguments passed to logger constructor
        
    Returns:
        Logger instance
    """
    if logger_type.lower() == 'tensorboard':
        return TensorBoardLogger(**kwargs)
    elif logger_type.lower() == 'wandb':
        return WandBLogger(**kwargs)
    elif logger_type.lower() == 'none':
        return DummyLogger()
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")


# Legacy compatibility aliases
TensorBoardLogger = TensorBoardLogger
WandBLogger = WandBLogger
