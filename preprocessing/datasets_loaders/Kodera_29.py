from ..ETL import ETL
from .data_formats.Saleh_format import SalehFormat
from .data_formats.Mochura_format import MochaFormat

import warnings

import os
import mne
from mne.io import Raw


class Kodera(ETL):

    data_formats = [MochaFormat, SalehFormat]

    def __init__(self):
        self.dataset_name = "kodera_29"
        self.base_path = self.data_path + self.dataset_name

    def load_data(self):
        processed_files = []
        for file in os.listdir(self.base_path):
            file_path = self.base_path + "/" + file
            for data_file in os.listdir(file_path):
                print(data_file)
                if data_file.endswith(".vhdr"):
                    for data_format in self.data_formats:
                        processed_file = data_format.create_file(file_path=file_path + "/" + data_file, file_name=data_file)
                        if processed_file is not None:
                            processed_file.read_raw()
                            processed_files.append(processed_file)
                            break
        # Group files belonging to the same subject in a list
        files_per_subject = []
        for i in range(len(processed_files)):
            file = processed_files[i]
            # This file already belongs to a subject which has been processed
            if file is None:
                continue

            subject_files = [file]
            for j in range(i + 1, len(processed_files)):
                other_file = processed_files[j]
                if file.same_subject(other_file):
                    subject_files.append(other_file)
                    processed_files[j] = None  # Mark the file as processed

            processed_files[i] = None  # Mark the file as processed
            files_per_subject.append(subject_files)

        return files_per_subject






