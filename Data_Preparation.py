import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from Data_Preprocessing import Preprocessor
from Data_Validation import Validate
from Data_Transformation import Transformer
from Load_data import Get_Data

class Feature_Engineering:
    """
    The Feature Engineering class is hard coded as minimal as possible based on the data recieved.
    Written By : Sabtha Rishi
    Version : 1.0
    
    """

    def __init__(self,filepath):
      self.filepath = filepath
      self.data = Get_Data().load_from_folder(self.filepath)
      self.preprocessor = Preprocessor()
      self.transformer = Transformer()
        

    def prepare_data(self):

      try:

        self.data = Validate().validate_train(self.data)

        self.df = self.data.copy()
        self.df = self.transformer.remove_missing_values(self.df)
        self.df["month"] = self.preprocessor.change_datatype(self.df,column = "month",to_datatype="int")

        self.df["started_at"] = self.preprocessor.convert_datetime(data=self.df, columns="started_at")
        self.df["ended_at"] = self.preprocessor.convert_datetime(data=self.df, columns="ended_at")


        self.df["distance"] = self.preprocessor.lat_lon_distance(self.df,"end_lng", "start_lng", "end_lat", "start_lat")

        self.df["start_station_name_fre"] = self.preprocessor.map_value_counts(self.df,"start_station_name")
        self.df["end_station_name_fre"] = self.preprocessor.map_value_counts(self.df,"end_station_name") 

        self.df["start_date"] = self.df["started_at"].apply(lambda x: x.day) 

        self.df["travel_dur"] = self.df["ended_at"] - self.df["started_at"]
        self.df["travel_dur"] = self.df["travel_dur"].apply(lambda x: x.seconds/3600)

          
        self.df["start_daytime"] = self.df["started_at"].apply(lambda x: x.time().hour)

        self.df['start_dayOfWeek'] = self.df['started_at'].dt.weekday

        self.cal = calendar()
        self.dr = pd.date_range(start=self.df["started_at"].min(), end=self.df["started_at"].max())
        self.holidays = self.cal.holidays(start=self.dr.min(), end=self.dr.max())
        self.df['is_holiday'] = self.df['started_at'].dt.date.astype('datetime64').isin(self.holidays)

        self.X_feat = ["rideable_type", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual", "month", "distance", "start_station_name_fre", "end_station_name_fre",
        "start_date", "start_daytime", "start_dayOfWeek", "is_holiday", "travel_dur"]

        self.df = self.df[self.X_feat]
        self.df['is_member'] = self.df['member_casual'].apply(lambda x: 1 if x =='member' else 0)
        self.df['is_holiday'] = self.df['member_casual'].apply(lambda x: 1 if x == True else 0)

        self.df["is_electric_bike"] = self.df['rideable_type'].apply(lambda x: 1 if x =='electric_bike' else 0)
        self.df["is_classic_bike"] = self.df['rideable_type'].apply(lambda x: 1 if x =='classic_bike' else 0)
        self.df["is_docked_bike"] = self.df['rideable_type'].apply(lambda x: 1 if x =='docked_bike' else 0)

        self.df = self.df.drop(['rideable_type','member_casual'],axis=1)

        self.train, self.val, self.test = self.preprocessor.split_data(self.df)

        self.preprocessor.export_data(self.train,"Prepared_Data"+"/"+"train.csv")
        self.preprocessor.export_data(self.val,"Prepared_Data"+"/"+"val.csv")
        self.preprocessor.export_data(self.test,"Prepared_Data"+"/"+"test.csv")

        self.x_train , self.y_train = self.preprocessor.X_y_split(self.train,"travel_dur")
        self.x_val , self.y_val = self.preprocessor.X_y_split(self.val,"travel_dur")
        self.x_test , self.y_test = self.preprocessor.X_y_split(self.test,"travel_dur")

        self.x_train, self.x_val, self.x_test, self.scaler = self.preprocessor.scale_data(self.x_train, self.x_val, self.x_test)

        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, self.scaler

      except Exception:
        raise Exception

        


