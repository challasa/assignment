import logging.config
import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] 
CONFIG_FILE = PROJECT_ROOT / "config.json"

def setup_logging(logger_name: str = "generic") -> logging.Logger:
    """
    Sets up logging using dictConfig from the config.json file.

    Params:
    -------
    logger_name: name of the logger as provided in config.json

    Returns:
    -------
    Logger: logger object with set handlers and formatters is returned
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # apply the configuration from config.json
        logging.config.dictConfig(config)
        return logging.getLogger(logger_name)
    except Exception as e:
        print(f"Error loading logging configuration: {e}")
        # fallback to basic setup on failure
        logging.basicConfig(level=logging.WARNING, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(logger_name)