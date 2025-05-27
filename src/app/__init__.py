from src.logs import setup_logger, logger

# Initialize logging when the package is imported
setup_logger()

__all__ = ["logger", "setup_logger"]
