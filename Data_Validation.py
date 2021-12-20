class Validate:
    """
    This class validates both the Training Batch Data and Prediction Data

    """

    def __init__(self):
        pass
        
    def validate_train(self,data):   

        """
        The validate_train method checks the number of columns, Column Names and the Datatype of each columns
        Returns : The data if validation is successful
        Validation Unsuccessful : Returns None and Prints a message
        """

        self.data = data
        if self.data.shape[1] == 14:
            actual_columns = ['ride_id','rideable_type','started_at','ended_at','start_station_name','start_station_id','end_station_name','end_station_id','start_lat','start_lng','end_lat','end_lng','member_casual','month']
            if (list(self.data.columns)) == actual_columns:
                object_dtype_cols = ['ride_id','rideable_type','started_at','ended_at','start_station_name','start_station_id','end_station_name','end_station_id','member_casual','month']
                float_dtype_cols = ['start_lat', 'start_lng', 'end_lat', 'end_lng']
                if (list(self.data.select_dtypes("object"))) == object_dtype_cols:
                    if (list(self.data.select_dtypes("float"))) == float_dtype_cols:
                        print("Validation Successfull")
                        return self.data               
        else:
            print("Validation Failed : Check the Training Data")