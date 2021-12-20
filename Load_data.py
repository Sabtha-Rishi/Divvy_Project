import pandas as pd
import os

class Get_Data:
    """
     This class is used to load the data to a pandas dataframe

    """

    def __init__ (self):
        pass

    def load_csv(self, file_path):


        """
        Use this to load a single file

        """
        self.file_path = file_path
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            raise Exception

    def load_from_folder(self, file_path):

        """
        Use this method to load all the files in the folder

        """

        self.file_path = file_path
        try:
            all_files = []
            files = os.listdir(self.file_path)
            for file in files:
                file_path = self.file_path +'/'+ file
                file_df = pd.read_csv(file_path)
                file_df["month"] = file.split(sep="-")[0][-2:]
                all_files.append(file_df)
            print("All files in the folders has been loaded")
            return pd.concat(all_files).dropna().reset_index(drop=True)
        
        except Exception as e:
            print("Files cannot be loaded from the folder")
            raise Exception


