import librosa
import logging
import os


def load_audio(path , sr=22050):

    if not isinstance(path,str) or not path.strip():
        logging.error("Invalid audio path.")
        return None, None 
    
    if not os.path.exists(path):
        logging.error(f"Audio file not found: {path}")
        return None , None
    
    audio,sr=librosa.load(path,sr=sr)

    return audio , sr


def extract_mel_spectrogram(audio,sr,n_mels=128):

    if audio is None:
        logging.error("Audio is None")
        return None
    
    if sr is None:
        logging.error("Sr is None")
        return None 

    S=librosa.feature.melspectrogram(audio,sr=sr,n_mels=n_mels)

    S_db=librosa.power_to_db(S)

    return S_db

