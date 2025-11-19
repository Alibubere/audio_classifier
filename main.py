import logging
import os
import numpy as np
from src.data_loader import load_metadata, load_dataset
from src.preprocessing import prepare_dataset
from src.model import build_model
from src.train import train_model


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
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("logging setup successfully")


def main():
    setup_logging()
    metadata_df = load_metadata("data/UrbanSound8K/metadata/UrbanSound8K.csv")

    X, y = load_dataset("data/UrbanSound8K/audio", metadata_df)

    X_processed, y_processed = prepare_dataset(X, y, max_len=256)

    input_shape = (X_processed.shape[1], X_processed.shape[2], 1)
    num_classes = len(np.unique(y_processed))
    model = build_model(input_shape, num_classes)

    history, model = train_model(model, X_processed, y_processed)

    os.makedirs("models",exist_ok=True)
    
    model.save("models/audio_classifier_model.h5")

    logging.info("Model saved successfully")


if __name__ == "__main__":
    main()
