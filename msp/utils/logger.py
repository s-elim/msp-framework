import logging
import os
import sys
from typing import Any, Dict, Optional

# Attempt to import wandb, but do not crash if it is not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Constants for log formatting
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "msp",
    output_dir: Optional[str] = None,
    rank: int = 0,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns the core Python logger.

    Sets up a stream handler for the console. If output_dir is provided and 
    the process is the primary node (rank == 0), it also adds a file handler.

    Args:
        name (str): Name of the logger. Defaults to "msp".
        output_dir (Optional[str]): Directory to save the local log file.
        rank (int): Process rank for Distributed Data Parallel (DDP). 
                    Only rank 0 logs to avoid multi-GPU terminal spam.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured python logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging in some environments

    # Clear existing handlers to prevent duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Handler (Only print on master node)
    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 2. File Handler (Only write on master node if dir is provided)
    if rank == 0 and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "experiment.log"))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class LoggerManager:
    """
    A unified manager for handling both text logs and experiment metrics.

    Wraps the standard python logger and Weights & Biases (wandb).
    Ensures safe degradation if wandb is disabled or unavailable.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        use_wandb: bool = True,
        output_dir: Optional[str] = None,
        rank: int = 0,
    ) -> None:
        """
        Initializes the LoggerManager.

        Args:
            project_name (str): WandB project name.
            experiment_name (str): Specific run name.
            config (Dict[str, Any]): Hyperparameter configuration dict to log.
            use_wandb (bool): Whether to attempt syncing metrics to WandB.
            output_dir (Optional[str]): Path to save local logs and artifacts.
            rank (int): DDP process rank. Only rank 0 interacts with WandB.
        """
        self.rank = rank
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE and (self.rank == 0)

        # Setup standard Python logger
        self.logger = setup_logger(output_dir=output_dir, rank=rank)

        # Initialize WandB
        if self.use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    dir=output_dir,
                    resume="allow",
                )
                self.logger.info("Successfully initialized Weights & Biases.")
            except Exception as e:
                self.logger.error(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
        elif use_wandb and not WANDB_AVAILABLE and rank == 0:
            self.logger.warning(
                "WandB requested but not installed. Run `pip install wandb`. "
                "Falling back to local logging only."
            )

    def info(self, msg: str) -> None:
        """Log an INFO level message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a WARNING level message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an ERROR level message."""
        self.logger.error(msg)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Logs numerical metrics (e.g., VIB distortion, rate, coverage).

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and values.
            step (Optional[int]): The current global training step or epoch.
        """
        if self.rank != 0:
            return

        # Always log a summary to the console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        step_prefix = f"Step {step} | " if step is not None else ""
        self.logger.info(f"{step_prefix}Metrics -> {metrics_str}")

        # Sync to WandB if active
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_artifact(self, file_path: str, artifact_type: str = "model") -> None:
        """
        Saves a file to WandB as an artifact (e.g., a PyTorch checkpoint).

        Args:
            file_path (str): The local path to the file.
            artifact_type (str): The type category for the artifact.
        """
        if self.rank != 0:
            return

        if not os.path.exists(file_path):
            self.logger.error(f"Artifact not found: {file_path}")
            return

        if self.use_wandb:
            artifact_name = os.path.basename(file_path)
            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
            self.logger.info(f"Logged artifact '{artifact_name}' to WandB.")
        else:
            self.logger.info(f"Local artifact saved at: {file_path} (WandB disabled)")

    def close(self) -> None:
        """Closes connections and finishes the run gracefully."""
        if self.use_wandb:
            wandb.finish()
            self.logger.info("Closed Weights & Biases connection.")
