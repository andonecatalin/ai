
import psycopg2
import numpy as np

def create_query(name, table_name):
    replacement_map=[f'{name}_Open',f'{name}_High',f'{name}_Low',f'{name}_Close',f'{name}_Adj_Close',f'{name}_Volume','t']
    query=f"SELECT {', '.join(replacement_map)} FROM {table_name} WHERE {replacement_map[0]} IS NOT NULL ORDER BY t ASC"
    return query

def request_data(name,table_name,dbname,dbuser,dbpassword,dbhost,dbport):
    conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)
    cursor=conn.cursor()
    query=create_query(name,table_name)
    cursor.execute(query)
    data=cursor.fetchall()
    conn.close()

    return data


def normal(matrice, maxim):
    max=np.max(matrice)
    min=np.min(matrice)
    for i in range(len(matrice)):
        matrice[i]=(matrice[i]-min)/(max-min)*maxim
    return matrice

def graph(timp,data):
    timp=normal(timp,256)
    data=normal(data,256)
    p=np.zeros((len(timp),len(timp)))
    for i in range(len(timp)):
        p[i][round(data[i])]=1
    return p
