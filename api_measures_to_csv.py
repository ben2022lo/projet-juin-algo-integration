import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import logging
import json
import requests


class GetData:
    def __init__(self):
        j_file = open("config_api.json")
        id = json.load(j_file)
        self.id = id
        j_file.close()

    def get_data_api(self, c_id, tm_id, t0, t1):
        url = self.id["api_url"] + "Mesures"
        print(url)
        response = requests.get(
            self.id["api_url"] + "Mesures" + "/" + str(c_id) + "/" + str(tm_id) + "/" + t0 + "/" + t1)

        if response.status_code == 200:
            print("measurements sucessfully recover")
            return response.json()
        else:

            # logging.basicConfig(filename=self.id["logs"], level=logging.INFO,
            #                    format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(
                "could not get measurements table. code error :{} message : {} ".format(response.status_code,
                                                                                        response.text))
            return False


# info to get data
c_id = 1
tm_id = 1
t0 = "2014-04-01 00:00:00"
t1 = "2014-04-14 23:55:00"

get = GetData()
data = get.get_data_api(c_id, tm_id, t0, t1)


df = pd.DataFrame(data)
csv_file = "api_measure_test.csv"
df.to_csv(csv_file, index=False)

# df = pd.read_csv(csv_file)
# plt.plot(df.horodate, df.valeur)
# plt.show()

