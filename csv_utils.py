import os
import tarfile
import urllib.request
import zipfile

import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CSVDataPreprocess():

    def __init__(self):
        pass

    def download_dataset(self, dataset_name):
        if dataset_name.lower() == 'fetal_health':
            DOWNLOAD_ROOT = "https://raw.githubusercontent.com/xuwayyy/F21DL_Datasets_location/main/"
            DATASET_PATH = os.path.join("datasets")
            DATASET_URL = DOWNLOAD_ROOT + "fetal_health.tar"

            os.makedirs(DATASET_PATH, exist_ok=True)
            tgz_path = os.path.join(DATASET_PATH, "fetal_health.tgz")
            urllib.request.urlretrieve(DATASET_URL, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=DATASET_PATH)
            housing_tgz.close()
        elif dataset_name.lower() == 'medical_mnist':
            url = 'https://github.com/xuwayyy/F21DL_Datasets_location/raw/main/Medical%20MNIST.zip'
            zip_file_path = 'Medical_MNIST.zip'

            response = requests.get(url)
            if response.status_code == 200:
                with open(zip_file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {zip_file_path} from GitHub.")
            else:
                print(f"Failed to download file from {url}. Status code: {response.status_code}")

            extract_path = 'Medical_MNIST_unzipped'
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                print(f"Unzipped the contents to {extract_path}")

    def standard_normalization(self, X):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X)
        return scaled_data

    def min_max_normalization(self, X):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(X)
        return scaled_data
