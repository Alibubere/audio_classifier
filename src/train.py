import logging
import numpy as np
from sklearn.model_selection import train_test_split


def train_model(model, X, y, epochs=20, batch_size=32, val_split=0.2):

    if model is None:
        logging.error("Invalid Model")
        return None, None

    if X is None or y is None:
        logging.error("X or y is None")
        return None, None

    try:

        logging.info("Starting model training ....")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )

        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )
        logging.info("Model trained successfully")
        return history, model

    except Exception as e:
        logging.exception(f"failed to train {e}")
        return None, None
