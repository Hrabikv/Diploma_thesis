from src.enums.MovementType import MovementType
from .File_format import FileFormat

import re


class MochaFormat(FileFormat):
    file_name_pattern = r"(\d+)([mz])(\d{8})([rl]h)(\d+).vhdr"

    def __init__(self, file_path: str, ID: int, gender: int, date: str, movement_type: MovementType,
                 trial_order_number: int):
        super().__init__()
        self._file_path = file_path
        self.ID = ID
        self.gender = gender
        self.date = date
        self._movement_type = movement_type
        self.trial_order_number = trial_order_number

    @staticmethod
    def create_file(file_path, file_name):
        matched = re.match(MochaFormat.file_name_pattern, file_name)
        if not matched:
            return None
        ID = matched.group(1)
        gender = matched.group(2)
        date = matched.group(3)
        movement_type = MovementType.get_type(matched.group(4))
        trial_order_number = matched.group(5)

        return MochaFormat(file_path, ID, gender, date, movement_type, trial_order_number)

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
        return self.gender == other.gender and self.ID == other.ID and self.date == other.date

