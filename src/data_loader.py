import pandas as pd
import logging
import os


def load_metadata(csv_path):

    if not isinstance(csv_path, str) or csv_path.strip():
        logging.error("Invalid CSV path")
        return None

    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found {csv_path}")
        return None

    df = pd.read_csv(f"{csv_path}/UrbanSound8K.csv")

    return df
