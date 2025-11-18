import logging
import numpy as np


def pad_spectrogram(spec, max_len=256):

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

    if spec is None:
        logging.error("Spec is None")
        return None

    min_val = spec.min()
    max_val = spec.max()

    if max_val - min_val == 0:
        logging.warning("Zero variance in spectrogram")
        return spec

    return (spec - min_val) / (max_val - min_val)

