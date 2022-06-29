import pandas as pd
import numpy as np


class Loader:
    """
    Load csv file into memory.
    Expected columns in csv: SessionId (String), ItemId (int64), Time (seconds)
    """

    @staticmethod
    def read_csv(full_path_to_csv_file):
        df = pd.read_csv(full_path_to_csv_file, sep='\t',
                           dtype={'SessionId': object, 'ItemId': 'str', 'Time': 'float64'}) #ItemID from int64 to string
        df['Time'] = df['Time'].apply(lambda x: int(round(x, 0)))
        return df