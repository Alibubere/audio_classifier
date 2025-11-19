import logging
import os
from src.data_loader import load_metadata, load_dataset
from src.preprocessing import prepare_dataset


def setup_logging():
    """
    Set Up logging for the Entire Pipeline
    """

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=([logging.FileHandler(log_file), logging.StreamHandler()],),
    )
    logging.info("logging setup successfully")


def main():
    setup_logging()
    metadata_df = load_metadata("data/UrbanSound8K/metadata/UrbanSound8K.csv")

    X , y = load_dataset("data/UrbanSound8k/audio",metadata_df)

    X_processed , y_processed = prepare_dataset(X , y ,max_len=256)



if __name__ == "__main__":
    main()
