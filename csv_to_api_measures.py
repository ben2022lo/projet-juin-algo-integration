import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import logging
import json
import requests


class InsertData:
    def __init__(self):
        j_file = open("config_put_data.json")
        id = json.load(j_file)
        self.url = id['api_url']
        self.csv = id['csv_dir']
        j_file.close()
        self.token = ""

    def get_token(self):
        request_body = {"grant_type": None, "username": "flutter@halias.fr", "password": "azer123", "scope": "",
                        "client_id": None, "client_secret": None}
        response = requests.post(self.url + "token", data=request_body)
        self.token = response.json()['access_token']

    def insert_data_api(self, data_to_insert):
        #headers = {'Authorization': 'Bearer ' + self.token}
        response = requests.post(self.url + "mesures", json=data_to_insert)

        if response.status_code == 200:
            print("measurements sucessfully inserted")
            return True
        else:

            logging.error(
                "could not insert in measurements table. code error :{} message : {} ".format(response.status_code,
                                                                                              response.text))
            return False


if __name__ == "__main__":
    insert = InsertData()

    # Get all the csv files of insert.csv (=csv dir)
    csv_files = [f for f in os.listdir(insert.csv) if ".csv"]
    data = []
    for f in csv_files:
        file_path = os.path.join(insert.csv, f)
        file_data = pd.read_csv(file_path)
        data.append([file_data, f])

    for df, name in data:
        liste = []
        for i in df.index:
            list_data = [1, 1, str(df['timestamp'][i]), None, df['value'][i], 1]
            liste.append(list_data)
        data_to_insert = {"list_mesures": liste}
        # insert.get_token()
        insert.insert_data_api(data_to_insert)
