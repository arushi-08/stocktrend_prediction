
import os
import pandas as pd

import gdown

datasets = {
    "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests": "1QUlr76Nn_wdUsZtUIg5RTZ981sRnZJJa"
    
}

hashes = {
    "1QUlr76Nn_wdUsZtUIg5RTZ981sRnZJJa": "bc3f451f02a32f61ab93e2d7dcaafa633702a83e91c74220ecf4145abbeb2e30"
}

def get_path(dataset_name):
    return f"datasets/{dataset_name}/"

def read_files(path, file_names=[]):
    file_data = {}
    if not file_names:
        file_names = os.listdir(path)
    for f in file_names:
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            if f.endswith('.csv'):
                file_data[f] = pd.read_csv(fp)
    return file_data

def load_dataset(dataset_name, file_name=None, file_names=[]):
    dataset_id = datasets.get(dataset_name)
    dataset_path = get_path(dataset_name)
    archive_path = f"{dataset_path}/archive.zip"
    shasum = f"sha256:{hashes.get(dataset_id)}"
    gdown.cached_download(id=dataset_id, path=archive_path ,postprocess=gdown.extractall, hash=shasum)
    
    if file_name is not None:
        file_names = []
    file_data = read_files(dataset_path, file_names)

    if file_name is None:
        return file_data
    else:
        return file_data.get(file_name, {})


if __name__ == "__main__":
    for dataset_name in datasets:
        datasets = load_dataset(dataset_name)
        print(datasets)