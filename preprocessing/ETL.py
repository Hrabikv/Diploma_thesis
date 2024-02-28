from abc import ABC, abstractmethod


class ETL(ABC):
    data_path = "data/"

    @abstractmethod
    def load_data(self):
        pass


