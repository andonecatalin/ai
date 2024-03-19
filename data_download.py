#import json
import requests
import psycopg2
import pandas as pd

start_date="2006-01-01"
end_date="2023-12-31"

dbname='ai_dataset'
dbuser='postgres'
dbpassword='parola'
dbport=5432
dbhost='127.0.0.1'
conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)


def replace_keys_in_json_files(data, name):
  """Replaces keys in multiple JSON files based on a key mapping.

  Args:
    filenames: A list of file paths to the JSON files.
    key_map: A dictionary mapping old keys to their new replacements.
"""
  char_to_replacement_map={'o':f'{name}_Open','h':f'{name}_High', 'l':f'{name}_Low', 'c':f'{name}_Close', 'a':f'{name}_Adj Close', 'v':f'{name}_Volume'}

      # Iterate through the list and replace keys within dictionaries
  for i, item in enumerate(data):
    if isinstance(item, dict):  # Check if the item is a dictionary
        data[i] = {char_to_replacement_map.get(key, key): value for key, value in item.items()}
    else: 
       print('Error in key replacement')
          # Handle non-dictionary items (you can choose to skip them, raise an error, etc.)
  return data      
def tickers2(tickers):
    replaced=[]
    for i in range(len(tickers)): 
        replaced.append(tickers[i].replace('.','_'))
    return replaced

#api_key='dd75440b5984491b9f3593d8cc275ed4'
api_key='348c590f9b0248638d37b8f381287cf1'
data_json=[]
tickers=["AAPL","MSFT",'SPY','XAUUSD.OANDA','BCO.ICMTRADER']
x=tickers2(tickers)
concatanated_data=[]
for i in range(len(tickers)):
    try:
        response=requests.get(f'https://api.darqube.com/data-api/market-data/historical/daily/{tickers[i]}?token={api_key}&start_date={start_date}&end_date={end_date}')
        response.raise_for_status()
        data_json=response.json()
        data_json=replace_keys_in_json_files(data_json,x[i])
        concatanated_data.append(data_json)
      
          
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")

table_name='tabela'
cursor = conn.cursor()

for lists in concatanated_data:
  first_iteration=True
  for sub_list in lists:

    if first_iteration:
      keys = sub_list.keys()
     
      keys=[item.replace(' ','_') for item in keys]
      #print(keys)
      # Create table dynamically based on the keys
      column_definitions = ", ".join([f"ADD COLUMN IF NOT EXISTS {key} REAL " for key in keys])
      create_table_query = f"""
          ALTER TABLE {table_name}
          {column_definitions}
            ;
      """
      cursor.execute(create_table_query)

      #print(column_definitions)
    first_iteration=False

    
    #cursor = conn.cursor()
    
    
    value=list(sub_list.values())
  
    # Prepare the INSERT query dynamically
    insert_query = f"""
      INSERT INTO {table_name} ({", ".join(keys)})
      VALUES({[number for number in value]})"""
    insert_query=insert_query.replace('[','')
    insert_query=insert_query.replace(']','')
    cursor.execute(insert_query)
    # Insert data row by row
    #print(insert_query)

      #conn.commit()
conn.commit()
conn.close()