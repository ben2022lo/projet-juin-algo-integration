import pandas as pd
import logging
import json
import requests
from datetime import datetime

class GetData:
    def __init__(self):
        j_file = open("config_get_data.json")
        id = json.load(j_file)
        self.url = id['api_url']
        self.c_id = id['c_id']
        self.tm_id = id['tm_id']
        self.t0 = id['t0']
        self.t1 = id['t1']
        j_file.close()

    def get_data_api(self):
        url = self.url + "Mesures" + "/" + str(self.c_id) + "/" + str(self.tm_id) + "/" + self.t0 + "/" + self.t1
        response = requests.get(url)

        if response.status_code == 200:
            print("measurements sucessfully recover")
            return response.json()
        else:
            logging.error(
                "could not get measurements table. code error :{} message : {} ".format(response.status_code,
                                                                                        response.text))
            return False


if __name__ == "__main__":
    get = GetData()
    data = get.get_data_api()

    df = pd.DataFrame(data)
    df['horodate'] = pd.to_datetime(df['horodate'])
    df = df.drop_duplicates(subset=['horodate'])
    df = df.reset_index(drop=True)
    df['horodate'] = df['horodate'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    t0 = datetime.strptime(get.t0, '%Y-%m-%d %H:%M:%S')
    t1 = datetime.strptime(get.t1, '%Y-%m-%d %H:%M:%S')
    t0_formatted = t0.strftime('%Y-%m-%d_%H-%M-%S')
    t1_formatted = t1.strftime('%Y-%m-%d_%H-%M-%S')
    csv_file = "api_measure_" + str(get.tm_id) + "_" + str(get.c_id) + "_" + t0_formatted + "_" + t1_formatted + ".csv"
    df.to_csv(csv_file, index=False)
