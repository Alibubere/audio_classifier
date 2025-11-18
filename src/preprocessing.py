import logging
import numpy as np


def pad_spectrogram(spec, max_len=256):

    # ------------------------------------------------------
    # pad_spectrogram()
    # ------------------------------------------------------
    # Purpose:

    # Make sure every spectrogram has the same time length (max_len=256).
    # Neural networks don’t like variable-size inputs.
    
    if spec is None:
        logging.error("Spec is None")
        return None

    try:
        current_len = spec.shape[1]

        if current_len > max_len:
            return spec[:, :max_len]

        if current_len < max_len:
            padding = max_len - current_len
            return np.pad(spec, ((0, 0), (0, padding)), mode="constant")

        return spec

    except Exception as e:
        logging.exception(e)


def normalize_spectrogram(spec):

    # ------------------------------------------------------
    # normalize_spectrogram()
    # ------------------------------------------------------
    # Purpose:

    # Convert each spectrogram to values between 0 and 1.

    if spec is None:
        logging.error("Spec is None")
        return None

    min_val = spec.min()
    max_val = spec.max()

    if max_val - min_val == 0:
        logging.warning("Zero variance in spectrogram")
        return spec

    return (spec - min_val) / (max_val - min_val)


def prepare_dataset(X, y, max_len=256):
    # ------------------------------------------------------
    # prepare_dataset()
    # ------------------------------------------------------
    # Purpose:

    # Pad all spectrograms

    # Normalize them

    # Convert to numpy arrays

    # Pack labels
    if X is None or y is None:
        logging.error("X or y value is None")
        return None, None

    processed_X = []

    try:
        for spec in X:

            padded = pad_spectrogram(spec, max_len)

            if padded is None:
                logging.warning("Skipping invalid spectrogram")
                continue

            normalized = normalize_spectrogram(padded)

            if normalized is None:
                logging.warning("Skipping normalization failed")
                continue

            processed_X.append(normalized)

        processed_X = np.array(processed_X)
        processed_y = np.array(y)

        return processed_X, processed_y

    except Exception as e:
        logging.exception(e)
        return None, None
