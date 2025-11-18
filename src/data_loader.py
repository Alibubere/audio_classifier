import pandas as pd
import logging
import os
from src.feature_extractor import load_audio, extract_mel_spectrogram


def load_metadata(csv_path):

    if not isinstance(csv_path, str) or not csv_path.strip():
        logging.error("Invalid CSV path")
        return None

    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    return df


def load_dataset(data_dir, metadata_df):

    if not isinstance(data_dir, str) or not data_dir.strip():
        logging.error("Invalid data directory")
        return None, None

    if not os.path.exists(data_dir):
        logging.error(f"Data not found {data_dir}")
        return None, None

    if metadata_df is None:
        logging.error("Meta is None")
        return None, None

    required_cols = ["slice_file_name", "fold", "classID"]

    for cols in required_cols:
        if cols not in metadata_df.columns:
            logging.error(f"Missing required columns {cols}")
            return None, None

    X = []
    y = []

    for index, row in metadata_df.iterrows():
        file_name = row["slice_file_name"]
        fold = row["fold"]
        label = row["classID"]

        full_path = os.path.join(data_dir, f"fold{fold}", file_name)

        audio, sr = load_audio(full_path)

        if audio is None:
            logging.warning(f"Skipping file: {full_path}")
            continue

        s_db = extract_mel_spectrogram(audio, sr)

        if s_db is None:
            logging.warning(f"Failed to process spectrogram :{full_path}")
            continue

        X.append(s_db)
        y.append(label)

    return X, y
