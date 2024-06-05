import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import logging
import json
import requests


class InsertData:
    def __init__(self):
        j_file = open("config_api.json")
        id = json.load(j_file)
        self.id = id
        j_file.close()
        self.token = ""

    def get_token(self):
        request_body = {"grant_type": None, "username": "flutter@halias.fr", "password": "azer123", "scope": "",
                        "client_id": None, "client_secret": None}
        response = requests.post(self.id["api_url"] + "token", data=request_body)
        self.token = response.json()['access_token']

    def insert_data_api(self, data_to_insert):
        #headers = {'Authorization': 'Bearer ' + self.token}
        response = requests.post(self.id["api_url"] + "mesures", json=data_to_insert)

        if response.status_code == 200:
            print("measurements sucessfully inserted")
            return True
        else:

            # logging.basicConfig(filename=self.id["logs"], level=logging.INFO,
            #                    format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(
                "could not insert in measurements table. code error :{} message : {} ".format(response.status_code,
                                                                                              response.text))
            return False


csv_dir = r"C:\Users\louha\OneDrive - HALIAS\Documents\GitHub\NAB\data\artificialWithAnomaly"

csv_files = [f for f in os.listdir(csv_dir) if ".csv"]
data = []
for f in csv_files:
    file_path = os.path.join(csv_dir, f)
    file_data = pd.read_csv(file_path)
    data.append([file_data, f])

# for df, name in data:
#     liste = []
#     for i in df.index:
#         list_data = [1, 1, str(df['timestamp'][i]), None, df['value'][i], 1]
#         liste.append(list_data)
#     data_to_insert = {"list_mesures": liste}
#     insert = InsertData()
#     insert.insert_data_api(data_to_insert)

df, name = data[0]
liste = []
for i in df.index:
    list_data = [1, 1, str(df['timestamp'][i]), None, df['value'][i], 1]
    liste.append(list_data)
data_to_insert = {"list_mesures": liste}
print(data_to_insert)
insert = InsertData()
#insert.get_token()
insert.insert_data_api(data_to_insert)
