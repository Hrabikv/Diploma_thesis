from enum import Enum

from .EpochEvent import EpochEvent


class MovementType(Enum):
    UNKNOWN = ("", "", -1)
    RESTING = ("", "", EpochEvent.RESTING_MIDDLE)
    LEFT = ("lh", "leva", EpochEvent.MOVEMENT_START)
    RIGHT = ("rh", "prava", 6)

    def get_epoch_event(self) -> int:
        return self.value[2]

    @staticmethod
    def get_type(name: str) -> 'MovementType':
        for t in MovementType:
            if t.value[0] == name or t.value[1] == name:
                return t
        return MovementType.UNKNOWN
