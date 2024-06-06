import pandas as pd
import logging
import json
import requests


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
    csv_file = f"api_measure_{get.c_id}_{get.tm_id}_{get.t0}_{get.t1}.csv"
    df.to_csv(csv_file, index=False)
