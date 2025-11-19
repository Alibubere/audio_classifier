import logging
import os

def setup_logging():
    log_path= "logs"
    os.makedirs(log_path,exist_ok=True)
    log_file=os.path.join(log_path,"pipeline.log")

    logging.basicConfig(
        
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=(
            logging.FileHandler(log_file),
            logging.StreamHandler
        )
    )
    logging.info("logging setup successfully")