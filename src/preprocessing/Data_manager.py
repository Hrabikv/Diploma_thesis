import numpy as np
import os

from src.config import NUMBER_OF_CLASSES, FEATURE_VECTOR

DATA_FOLDER = "PREPROCESSED_DATA_FOLDER"
DATA_FOLDER_02 = DATA_FOLDER + "/" + FEATURE_VECTOR


def create_folders(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)


def exists(path: str) -> bool:
    return os.path.exists(path)


def save_preprocessed_data(data: np.ndarray, labels: np.ndarray) -> None:

    create_folders(DATA_FOLDER)
    create_folders(DATA_FOLDER_02)

    if NUMBER_OF_CLASSES == 3:
        labels_file_path = DATA_FOLDER_02 + "/labels_3.npy"
        data_file_path = DATA_FOLDER_02 + "/data_3.npy"
    else:
        labels_file_path = DATA_FOLDER_02 + "/labels_2.npy"
        data_file_path = DATA_FOLDER_02 + "/data_2.npy"

    np.save(data_file_path, data, allow_pickle=True)
    np.save(labels_file_path, labels, allow_pickle=True)


def load_preprocessed_data() -> tuple:

    if NUMBER_OF_CLASSES == 3:
        labels_file_path = DATA_FOLDER_02 + "/labels_3.npy"
        data_file_path = DATA_FOLDER_02 + "/data_3.npy"
    else:
        labels_file_path = DATA_FOLDER_02 + "/labels_2.npy"
        data_file_path = DATA_FOLDER_02 + "/data_2.npy"

    if not exists(data_file_path):
        return None, None

    if not exists(labels_file_path):
        return None, None

    data = np.load(data_file_path, allow_pickle=True)
    labels = np.load(labels_file_path, allow_pickle=True)

    return data, labels
