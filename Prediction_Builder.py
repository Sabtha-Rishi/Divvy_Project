import pandas as pd
from datetime import datetime
from math import sin, cos, sqrt, atan2
import holidays
import pickle
from Data_Preprocessing import  Preprocessor



class Prediction_Data:

    def __init__(self):
        self.start_station_counts = pickle.load(open("Dictionary Files\start_station_name_counts.pkl", "rb"))
        self.end_station_counts = pickle.load(open("Dictionary Files\end_station_name_counts.pkl", "rb"))

        self.start_station_lat = pickle.load(open("Dictionary Files\start_station_lat.pkl", "rb"))
        self.start_station_lon = pickle.load(open("Dictionary Files\start_station_lon.pkl", "rb"))
        self.end_station_lat = pickle.load(open("Dictionary Files\end_station_lat.pkl", "rb"))
        self.end_station_lon = pickle.load(open("Dictionary Files\end_station_lon.pkl", "rb"))

        self.holidays = holidays.US()


    def get_distance(self,lat1,lon1,lat2,lon2):

        R = 6373.0

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        self.distance = R * c

        return self.distance


    def generate_pred_X(self, input_dict):

        self.prediction_data = {}
        self.input_dict = input_dict  


        self.prediction_data["start_lat"] = self.start_station_lat['start_lat'][self.input_dict['start_station_name']]
        self.prediction_data["start_lon"] = self.start_station_lon['start_lng'][self.input_dict['start_station_name']]
        self.prediction_data["end_lat"] = self.end_station_lat['end_lat'][self.input_dict['end_station_name']]
        self.prediction_data["end_lon"] = self.end_station_lon['end_lng'][self.input_dict['end_station_name']] 
        self.prediction_data["month"] = datetime.now().month  
        self.prediction_data["distance"] = self.get_distance(self.prediction_data["start_lat"], self.prediction_data["start_lon"], self.prediction_data["end_lat"], self.prediction_data["end_lon"])
        self.prediction_data["start_station_name_fre"] = self.start_station_counts[self.input_dict['start_station_name']]
        self.prediction_data["end_station_name_fre"] = self.end_station_counts[self.input_dict['end_station_name']]
        self.prediction_data["start_date"] = datetime.now().day
        self.prediction_data["start_daytime"] = datetime.now().hour
 
                        
        self.prediction_data["start_dayOfWeek"] = datetime.today().weekday()
        self.prediction_data["is_holiday"] = int(datetime.now().date() in self.holidays)


        self.prediction_data["is_member"] = self.input_dict['is_member']


        self.prediction_data["rideable_type"] = self.input_dict['rideable_type']
        self.prediction_data = pd.DataFrame(pd.Series(self.prediction_data)).T
        self.prediction_data["is_electric_bike"] = self.prediction_data['rideable_type'].apply(lambda x: 1 if x =='electric_bike' else 0)
        self.prediction_data["is_classic_bike"] = self.prediction_data['rideable_type'].apply(lambda x: 1 if x =='classic_bike' else 0)
        self.prediction_data["is_docked_bike"] = self.prediction_data['rideable_type'].apply(lambda x: 1 if x =='docked_bike' else 0)     
        self.prediction_data = self.prediction_data.drop('rideable_type',axis=1)
        self.prediction_data["is_member"] = self.prediction_data["is_member"].apply(lambda x: 1 if x=='member' else 0)


        self.num_cols = ["start_lat", "start_lon", "end_lat", "end_lon", "distance", "month", "start_date", "start_dayOfWeek", "start_station_name_fre", "end_station_name_fre"]
        self.prediction_data[self.num_cols] = (self.prediction_data[self.num_cols]).astype('int64')

        return self.prediction_data.astype('float32')

