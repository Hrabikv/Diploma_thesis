from src.enums.MovementType import MovementType
from .File_format import FileFormat

import re


class SalehFormat(FileFormat):
    file_name_pattern = r"(.*)_(.*)_(\d+)_([sb]e?z?_vibrator[uy]_s_haptikou)_(prava|leva).vhdr"

    def __init__(self, file_path: str, measure_date: str, trial_order_number: int, has_vibrators: bool,
                 movement_type: MovementType):
        super().__init__()
        self._file_path = file_path
        self.measure_date = measure_date
        self.trial_order_number = trial_order_number
        self.has_vibrators = has_vibrators
        self._movement_type = movement_type

    @staticmethod
    def create_file(file_path, file_name):
        matched = re.match(SalehFormat.file_name_pattern, file_name)
        if not matched:
            return None

        measure_date = matched.group(2)
        trial_order_number = matched.group(3)
        has_vibrators = matched.group(4).startswith("s")
        movement_type = MovementType.get_type(matched.group(5))

        return SalehFormat(file_path, measure_date, trial_order_number, has_vibrators, movement_type)

    @property
    def movement_type(self) -> MovementType:
        return self._movement_type

    @movement_type.setter
    def movement_type(self, value: MovementType) -> None:
        self._movement_type = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

    def same_subject(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.measure_date == other.measure_date and self.trial_order_number == other.trial_order_number