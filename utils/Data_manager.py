import numpy as np
import os

DATA_FOLDER = "PREPROCESSED_DATA_FOLDER"


def create_folders(folder_path: str) -> None:
    os.makedirs(folder_path, exist_ok=True)


def exists(path: str) -> bool:
    return os.path.exists(path)


def save_preprocessed_data(data: np.ndarray, labels: np.ndarray) -> None:

    create_folders(DATA_FOLDER)

    data_file_path = DATA_FOLDER + "/data.npy"
    labels_file_path = DATA_FOLDER + "/labels.npy"

    np.save(data_file_path, data, allow_pickle=True)
    np.save(labels_file_path, labels, allow_pickle=True)


def load_preprocessed_data() -> tuple:

    data_file_path = DATA_FOLDER + "/data.npy"
    labels_file_path = DATA_FOLDER + "/labels.npy"

    if not exists(data_file_path):
        return None, None

    data = np.load(data_file_path, allow_pickle=True)
    labels = np.load(labels_file_path, allow_pickle=True)

    return data, labels
