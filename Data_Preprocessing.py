import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import to_datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Preprocessor:

    """
    This class can be used to clean and prepare the data for modeling

    Wriiten By : Sabtha Rishi
    Version : 1.0
    
    """

    def __init__(self):
        pass
        

    def convert_datetime(self,columns,data):

        """
        This method converts object features to Pandas Datetime
        
        """
        self.data = data
        try:
            return pd.to_datetime(self.data[columns])
            
        
        except Exception:
            raise Exception

    def remove_columns(self,columns,data):

        """
        This method is used to drop unwanted columns
        
        """

        self.data = data
        try:
            self.useful_data = self.data.drop(columns, axis=1)
            return self.useful_data
        except Exception as e:
            raise e
        
    def lat_lon_distance(self,data,lon2,lon1,lat2,lat1):

        """
        This method is used to calculate the distance by using the latitude and longitude of the start and destination location of
        Returns : Pandas Series object containing the distance
        
        """
        self.df = data
        R = 6373.0 # Radius of Earth (approx)

        dlon = self.df[lon2] - self.df[lon1]
        dlat = self.df[lat2] - self.df[lat1]

        a = np.sin(dlat / 2)**2 + np.cos(self.df[lat1]) * np.cos(self.df[lat2]) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c
        return distance

    def scale_data(self,train_x,val_x,test_x):

        """
        This method is used to scale the numerical data from 0 to 1 
        Returns: Scaled DataFrame
        
        """
        self.train = train_x
        self.val = val_x
        self.test = test_x
        self.scaler = MinMaxScaler()

        try:
            self.scaled_train = self.scaler.fit_transform(self.train.values)
            self.scaled_val = self.scaler.transform(self.val.values)
            self.scaled_test = self.scaler.transform(self.test.values)
            return self.scaled_train, self.scaled_val, self.scaled_test, self.scaler
        
        except Exception as e:
            raise e
        
    def get_daytime(self,x):

        """
        This method maps the daytime based on the time of the day
        
        """

        self.x = x

        if (4<x<7):
            self.x = 1
        elif(7<x<12):
            self.x= 2
        elif (12<x<15):
            self.x= 3
        elif (15<x<19):
            self.x= 4
        elif (19<x<23):
            self.x= 5
        elif (23<x<4):
            self.x= 6
        else:
            self.x= 0

        return self.x
    
    def one_hot_encoder(self,data,cat_columns, drop__first=True):

        """
        This method returns One Hot Encoded dataframe
        
        """
        self.data = data
        self.drop_first = drop__first
        try:
            self.encoded_df = pd.get_dummies(data = self.data, prefix=cat_columns, prefix_sep="_", columns=cat_columns, drop_first = self.drop_first )
            return self.encoded_df
        
        except Exception as e:
            raise e

    def change_datatype(self,data,column,to_datatype):

        """
        This method is used to convert the datatype of the column to a specified datatype
        
        """
        self.data = data
        try:
            return self.data[column].astype(to_datatype)
        except Exception as e:
            raise e

    def map_value_counts(self,data,column):

        """
        This method maps the value counts of a categorical column to a new column to

        """

        self.data = data
        try:
            self.column_dict = self.data[column].value_counts().to_dict()
            return self.data[column].map(self.column_dict)
             
        
        except Exception as e:
            raise e

    def split_data(self,data):

        try:
            self.data=data
            self.train,self.val_and_test = train_test_split(self.data, random_state=0, test_size=0.05)
            self.val, self.test = train_test_split(self.val_and_test, random_state=0, test_size=0.5)

            return self.train, self.val, self.test
        except Exception as e:
            raise e 

    def X_y_split(self,data,target_col):

        self.X = data.drop(target_col, axis=1)
        self.y = data[target_col]

        return self.X , self.y

    def export_data(self,data,location):
        self.data=data
        self.location = location

        try:
            self.data.to_csv(self.location, index=False)

        except Exception as e:
            raise e 




                
    

        




        


    



