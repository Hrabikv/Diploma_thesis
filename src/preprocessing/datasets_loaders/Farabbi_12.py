from .Extraction import Extraction
from src.utils.Constants import EPOCH_DROP_EQUALIZE, REJECTION_THRESHOLD, NUMBER_OF_CLASSES
from src.enums.MovementType import MovementType


import numpy as np
import os
import mne
from mne import Epochs


class Farabbi(Extraction):

    def __init__(self):
        self.dataset_name = "farabbi_12"
        self.base_path = self.data_path + self.dataset_name
        self.files_per_subject = None
        self.events_codes_with_desc = {
            1: [1010, "End Of Session"],
            2: [32769, "none"],
            3: [32770, "Experiment Stop"],
            4: [32775, "Baseline Start"],
            5: [32776, "Baseline Stop"],
            6: [33281, "Train"],
            7: [33282, "Beep"],
            8: [33283, "none"],
            9: [768, "Start of Trial, Trigger at t=0s"],
            10: [769, "class1, Left hand"],
            11: [770, "class2, Right hand"],
            12: [774, "none"],
            13: [780, "none"],
            14: [781, "Feedback"],
            15: [786, "Cross on Screen"],
            16: [800, "End Of Trial"],
            17: [898, "none"]
        }

    def load_data(self):

        files_per_subject = []
        for subject in os.listdir(self.base_path):
            if subject == "chanlocs.locs":
                continue
            subject_path = self.base_path + "/" + subject
            subject_files = []
            for session in os.listdir(subject_path):
                session_path = subject_path + "/" + session
                for state in os.listdir(session_path):
                    state_path = session_path + "/" + state
                    for file in os.listdir(state_path):
                        subject_files.append(mne.io.read_raw_gdf(state_path + "/" + file))
            files_per_subject.append(subject_files)

        self.files_per_subject = files_per_subject

    def preprocess_data(self, sampling_frequency):
        data = []
        labels = []
        for subject in self.files_per_subject:
            epochs, epoch_labels = self.get_epochs(subject, sampling_frequency)
            if epochs is None or epoch_labels is None:
                continue

            data.append(epochs.get_data())
            labels.append(epoch_labels)

        data = np.array(data, dtype=object)
        labels = np.array(labels, dtype=object)

        return data, labels

    def get_epochs(self, files, sample_frequency: int) \
            -> tuple[None, None] | tuple[Epochs, np.ndarray]:
        tmin = -3.5
        tmax = 0.5
        l_freq = 8
        h_freq = 30
        channels = ['Channel 14', 'Channel 13', 'Channel 15']

        raw = mne.concatenate_raws(files)

        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw,
                            tmin=tmin, tmax=tmax,
                            events=events,
                            picks=channels,
                            preload=True,
                            verbose=False,
                            event_repeated='drop',
                            baseline=(None, tmin + 0.5))
        # Cropping the time because when the sampling frequency is e.g. 500 and trying to get 1 sec epoch,
        # the constructor returns 501 samples, by calling crop with include_tmax set to False we get the expected 500
        # samples
        epochs.crop(include_tmax=False)
        epochs.resample(sample_frequency)
        epochs.filter(l_freq, h_freq, verbose=False)
        epochs.drop_bad(reject={'eeg': REJECTION_THRESHOLD}, verbose=False)

        self.equalize_epoch_events(epochs)

        return epochs, epochs.events[:, 2]

    def equalize_epoch_events(self, epochs: Epochs) -> None:
        events = epochs.events
        events_to_keep = []
        indices_to_drop = []
        for i in range(len(events)):
            keep = False
            marker = events[i][2]
            last_marker = None if len(events_to_keep) == 0 else events_to_keep[len(events_to_keep) - 1][2]

            # Only keep the resting epoch if there was a movement epoch between this epoch and the last resting epoch
            if marker == 12 or marker == 13:
                epochs.events[i][2] = 2
                keep = True
            # Only keep the first movement epoch between two resting epochs
            elif marker == 10:
                epochs.events[i][2] = 5
                keep = True
            elif marker == 11:
                epochs.events[i][2] = 6
                keep = True

            if keep:
                events_to_keep.append(events[i])
            else:
                indices_to_drop.append(i)

        epochs.event_id = {f"{MovementType.RESTING.get_epoch_event()}": int(MovementType.RESTING.get_epoch_event()),
                           f"{5}": 5,
                           f"{6}": 6
                           }
        epochs.drop(indices_to_drop, reason=EPOCH_DROP_EQUALIZE)

    def find_min_sampling_frequency(self, min_sampling_frequency: float) -> int:

        for subject_files in self.files_per_subject:
            for file in subject_files:
                if file.times is not None:
                    min_sampling_frequency = min(file.info['sfreq'], min_sampling_frequency)

        return min_sampling_frequency
