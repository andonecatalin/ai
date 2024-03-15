class getdata:
    def __init__(self,name,table_name,cursor,dbname,dbuser,dbpassword,dbport,dbhost):
        self.name=name
        self.table_name=table_name
        self.cursor=cursor
        import psycopg2
        import pandas as pd
        self.dbname=dbname
        self.dbuser=dbuser
        self.dbpassword=dbpassword
        self.dbport=dbport
        self.dbhost=dbhost
        try:
            self.conn=psycopg2.connect(dbname=self.dbname,user=self.dbuser,password=self.dbpassword,host=self.dbhost,port=self.dbport)
        except psycopg2.OperationalError as e:
            # Raise a custom exception with details
            raise ConnectionError(f"Could not connect to PostgreSQL database: {e}")
        self.cursor=self.conn.cursor()
    def __str__(self):
        return f'''Connection data is: 
            Name={self.dbname} 
            User={self.dbuser}
            Password={self.dbpassword}
            Port={self.dbport}
            Host={self.dbhost}
            Table name={self.table_name}
            ticker={self.name}'''
            
    def create_query(self):
        self.replacement_map=[f'{self.name}_Open',f'{self.name}_High',f'{self.name}_Low',f'{self.name}_Close',f'{self.name}_Adj_Close',f'{self.name}_Volume']
        self.query=f"SELECT {', '.join(self.replacement_map)} FROM {self.table_name} WHERE {self.replacement_map[0]} IS NOT NULL ORDER BY t ASC"
        return self.query

    def get_data_by_tickers(self):
        query=self.create_query()
        print(query)
        self.cursor.execute(query)
        self.data=self.pd.DataFrame(self.cursor.fetchall())
        return self.data
    