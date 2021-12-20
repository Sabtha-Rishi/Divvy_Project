import pandas as pd

class Transformer:
    """
    This class contains various basic Trandformation methods required for the data (Based on validation)

    """

    def __init__(self):
        pass


    def remove_missing_values(self,data):
        self.data = data
        try:
            self.data.dropna().reset_index(drop=True)
            return self.data
        except Exception:
            raise Exception
        

