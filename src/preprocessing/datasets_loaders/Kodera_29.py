from .Extraction import Extraction
from src.utils.Constants import EPOCH_DROP_HALF_RESTING, EPOCH_DROP_EQUALIZE, REJECTION_THRESHOLD
from .data_formats.Saleh_format import SalehFormat
from .data_formats.Mochura_format import MochaFormat
from .data_formats.File_format import FileFormat
from src.enums.MovementType import MovementType
from src.enums.EpochEvent import EpochEvent

import random
import logging as log
import numpy as np
import os
import mne
from mne import Epochs


class Kodera(Extraction):
    data_formats = [MochaFormat, SalehFormat]

    def __init__(self):
        self.dataset_name = "kodera_29"
        self.base_path = self.data_path + self.dataset_name
        self.files_per_subject = None

    def load_data(self):
        processed_files = []
        for file in os.listdir(self.base_path):
            file_path = self.base_path + "/" + file
            for data_file in os.listdir(file_path):
                print(data_file)
                if data_file.endswith(".vhdr"):
                    for data_format in self.data_formats:
                        processed_file = data_format.create_file(file_path=file_path + "/" + data_file,
                                                                 file_name=data_file)
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

        self.files_per_subject = files_per_subject

    def preprocess_data(self, sampling_frequency) -> tuple[np.ndarray, np.ndarray]:
        data = []
        labels = []
        classification_type = 1

        for i, subject_files in enumerate(self.files_per_subject):
            log.info(f"Gathering data for subject {i + 1}.")
            if classification_type == 1:
                left = [left for left in subject_files if left.movement_type is MovementType.LEFT]
                right = [right for right in subject_files if right.movement_type is MovementType.RIGHT]

                left_epochs, left_labels = self.get_epochs(left, MovementType.LEFT.get_epoch_event(),
                                                           sampling_frequency)
                right_epochs, right_labels = self.get_epochs(right, MovementType.RIGHT.get_epoch_event(),
                                                             sampling_frequency)

                if left_epochs is None or right_epochs is None:
                    continue

                # Dropping half of the epochs representing the resting state of the patient from each set, in order to
                # try to maintain a balanced overall dataset where 1/3 is resting 1/3 is left movement and 1/3 is right
                # movement, otherwise the resting state would be much larger than the movements
                self.drop_half_resting(left_epochs)
                left_labels = left_epochs.events[:, 2]
                self.drop_half_resting(right_epochs)
                right_labels = right_epochs.events[:, 2]

                left_data = left_epochs.get_data()
                right_data = right_epochs.get_data()

                data.append(np.concatenate((left_data, right_data)))
                labels.append(np.concatenate((left_labels, right_labels)))

            # elif classification_type == 0:
            #     epochs, epochs_labels = _get_epochs(subject_files, EpochEvent.MOVEMENT_START, sampling_frequency)
            #
            #     if epochs is None:
            #         continue
            #
            #     subject_epochs.append(epochs)
            #
            #     epochs_data = epochs.get_data()
            #
            #     data.append(epochs_data)
            #     labels.append(epochs_labels)

        data = np.array(data, dtype=object)
        labels = np.array(labels, dtype=object)

        return data, labels

    def drop_half_resting(self, epochs: Epochs) -> None:
        epoch_events = epochs.events[:, 2]
        resting_indices = [i for i, event in enumerate(epoch_events) if event == MovementType.RESTING.get_epoch_event()]

        amount_to_delete = len(resting_indices) // 2
        # Randomly pick half of the resting indices to drop
        resting_indices_to_remove = random.sample(resting_indices, amount_to_delete)
        epochs.drop(resting_indices_to_remove, reason=EPOCH_DROP_HALF_RESTING)

    def get_epochs(self, files: list[FileFormat], movement_event: int, sample_frequency: int) \
            -> tuple[None, None] | tuple[Epochs, np.ndarray]:
        raws = [file.raw for file in files if file.raw is not None]

        if not raws:
            return None, None
        tmin = -3.5
        tmax = 0.5
        l_freq = 8
        h_freq = 30
        channels = ['Cz', 'C3', 'C4']

        raw = mne.concatenate_raws(raws)

        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw,
                            tmin=tmin, tmax=tmax,
                            events=events,
                            picks=channels,
                            preload=True,
                            verbose=False,
                            baseline=(None, tmin + 0.5))
        # Cropping the time because when the sampling frequency is e.g. 500 and trying to get 1 sec epoch,
        # the constructor returns 501 samples, by calling crop with include_tmax set to False we get the expected 500
        # samples
        epochs.crop(include_tmax=False)
        epochs.resample(sample_frequency)
        epochs.filter(l_freq, h_freq, verbose=False)
        epochs.drop_bad(reject={'eeg': REJECTION_THRESHOLD}, verbose=False)

        self.equalize_epoch_events(epochs, movement_event)

        return epochs, epochs.events[:, 2]

    def equalize_epoch_events(self, epochs: Epochs, movement_start_side_marker: int) -> None:
        events = epochs.events
        events_to_keep = []
        indices_to_drop = []
        for i in range(len(events)):
            keep = False
            marker = events[i][2]
            last_marker = None if len(events_to_keep) == 0 else events_to_keep[len(events_to_keep) - 1][2]

            # Only keep the resting epoch if there was a movement epoch between this epoch and the last resting epoch
            if marker == EpochEvent.RESTING_MIDDLE and marker != last_marker:
                keep = True
            # Only keep the first movement epoch between two resting epochs
            elif (marker == EpochEvent.MOVEMENT_START or marker == EpochEvent.MOVEMENT_ADDITIONAL) and \
                    (last_marker == EpochEvent.RESTING_MIDDLE):
                epochs.events[i][2] = movement_start_side_marker
                keep = True

            if keep:
                events_to_keep.append(events[i])
            else:
                indices_to_drop.append(i)

        epochs.event_id = {f"{MovementType.RESTING.get_epoch_event()}": int(MovementType.RESTING.get_epoch_event()),
                           f"{movement_start_side_marker}": movement_start_side_marker}
        epochs.drop(indices_to_drop, reason=EPOCH_DROP_EQUALIZE)

    def find_min_sampling_frequency(self, min_sampling_frequency: float) -> int:

        for subject_files in self.files_per_subject:
            for file in subject_files:
                if file.raw is not None:
                    min_sampling_frequency = min(file.raw.info['sfreq'], min_sampling_frequency)

        return min_sampling_frequency
