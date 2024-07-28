
import psycopg2
import numpy as np
import torch
class Protected_execution:
    def __init__(self) -> None:
        pass
    def create_query(name:str, table_name:str):
        replacement_map=[f'{name}_Open',f'{name}_High',f'{name}_Low',f'{name}_Close',f'{name}_Adj_Close',f'{name}_Volume']
        query=f"SELECT {', '.join(replacement_map)} FROM {table_name} WHERE {replacement_map[0]} IS NOT NULL ORDER BY t ASC"
        return query

    '''def request_data(self,name,table_name,dbname,dbuser,dbpassword,dbhost,dbport):
        conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)
        cursor=conn.cursor()
        query=self.create_query(name,table_name)
        cursor.execute(query)
        data=cursor.fetchall()
        conn.close()

        return data'''


    '''def normal(arr):
        """
        Normalizes a 1D NumPy array to a range between 0 and 1.

        Args:
            arr: A 1D NumPy array containing the data to be normalized.

        Returns:
            A NumPy array containing the normalized data between 0 and 1.
        """
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)'''


    
    
    '''def ema(self,data:np.ndarray,perioada:int)->np.ndarray:
        """calculeaza Exponantial Moving aAverage"""
        ema=[data[0]]
        for i in range(1, len(data)):
            weight=2.718/(perioada+1)
            ema.append(data[i]*weight+ema[i-1]*(1-weight))
        return np.array(ema,dtype=np.float64)'''
    '''def macd(self,data:np.ndarray,fast_ema=12,slow_ema=26,signal_ema=7):
        """calculeaza Moving Avarege Convargence/Divergence"""
        short_ema=self.ema(data,fast_ema)
        long_ema=self.ema(data,slow_ema)
        macd=short_ema-long_ema
        signal=self.ema(macd,signal_ema)
        return macd[-1]-signal[-1]'''
    def insert_column(self,data,column_name,table_name,dbname,dbuser,dbpassword,dbhost,dbport):
        conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)
        cursor=conn.cursor()
        query=f"""
            INSERT INTO {table_name} ({column_name})
            VALUES({[number for number in data]})
        """
        cursor.execute(query)
        conn.close()
    def create_cursor(dbname,dbuser,dbpassword,dbport,dbhost):
        conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)
        cursor=conn.cursor
        return cursor,conn
    def fit_to_range_tensor(data, min_value=-5, max_value=5):
        """
        Fits a PyTorch tensor of numbers into a specified range.

        Args:
            data: The PyTorch tensor to fit.
            min_value: The minimum value in the desired range (inclusive).
            max_value: The maximum value in the desired range (inclusive).

        Returns:
            A new PyTorch tensor containing the fitted values.
        """
        if not data.numel():
            return torch.tensor([])  # Handle empty tensor

        # Find minimum and maximum values in the tensor
        data_min = torch.min(data)
        data_max = torch.max(data)

        # Handle constant data case (all elements have the same value)
        if data_min == data_max:
            return torch.full_like(data, min_value)

        # Calculate scaling factor
        scale = (max_value - min_value) / (data_max - data_min)

        # Fit the tensor using broadcasting
        fitted_data = min_value + scale * (data - data_min)

        # Clip values to the range (optional)
        clipped_data = torch.clamp(fitted_data, min_value, max_value)

        return clipped_data
    def tensor_shortner(tensor, requested_size):
        if isinstance(tensor, torch.Tensor):
            tensor_size=tensor.size()[0]
        elif isinstance(tensor, list):
            tensor_size=len(tensor)
        if tensor_size<requested_size:
            raise Exception("Requested size is bigger or equal to tensor")
        
        i=tensor_size
        while i!=0:
            if i % requested_size==0:
                return tensor[:i]
            i-=1
    def image_builder(tensor:torch.Tensor, batch_size:int,shuffle=False):
        #makes consecutive batches in form of a list
        tensor_size=tensor.size()[0]
        batches=[]
        for i in range(tensor_size):
            if i+batch_size<tensor_size-1:
                batches.append([tensor[i:i+batch_size].tolist()])
        if shuffle:
            random.shuffle(batches)
            
        return batches

    
    def change(tensor:torch.Tensor, batch_size=32):
        #shape from image_builder function goes like this:
        #[batch number][leftover dimmension][element from batch]
        empty_list=[]
        for i in range(len(tensor)-1):
            first=tensor[i+1][0][0]
            last=tensor[i][0][batch_size-1]
            diffrence=last-first
            average=(last+first)/2
            percent_diffrence=(diffrence/average)*100*4
            #limits to 25 points +/- to make training achivable
            #times 4 to increase sensitivity
            if percent_diffrence >25:
                percent_diffrence=25
            if percent_diffrence<-25:
                percent_diffrence=-25
            empty_list.append(round(percent_diffrence))
        return empty_list        
    