import pandas as pd

def create_query(name,table_name):
    replacement_map=[f'{name}_Open',f'{name}_High',f'{name}_Low',f'{name}_Close',f'{name}_Adj_Close',f'{name}_Volume']
    query=f"SELECT {', '.join(replacement_map)} FROM {table_name} WHERE {replacement_map[0]} IS NOT NULL ORDER BY t ASC"
    return query

def get_data_by_tickers(nume,cursor):
    query=create_query(nume)
    print(query)
    cursor.execute(query)
    data=pd.DataFrame(cursor.fetchall())
    