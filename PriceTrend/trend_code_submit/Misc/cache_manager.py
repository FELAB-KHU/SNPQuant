import os.path as op

import pickle as pickle

import pandas as pd

from Misc import utilities as ut


class CacheManager(object):
    def __init__(self, root_dir: str):
        self.save_dir = ut.get_dir(root_dir)

    def get_save_path(self, file_name: str, file_type: str, subdir: str = None):

        if subdir is None:
            save_path = op.join(self.save_dir, f"{file_name}.{file_type}")
        else:
            save_dir = ut.get_dir(op.join(self.save_dir, subdir))
            save_path = op.join(save_dir, f"{file_name}.{file_type}")
        return save_path

    def save_to_cache(self, data, file_name: str, file_type: str, subdir: str = None):
        save_path = self.get_save_path(file_name, file_type, subdir)
        if file_type == "pkl":
            with open(save_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif file_type == "csv":
            data.to_csv(save_path)
        elif file_type == "feather":
            data.reset_index().to_feather(save_path)
        elif file_type == "pq":
            data.to_parquet(save_path)
        elif file_type == "tsv":
            data.to_csv(save_path, sep="\t", index=False, header=True)
        else:
            raise ValueError(f"{file_type} not supported")
        print(f"Successfully saved cache file to {save_path}")

    def load_from_cache(
        self, file_name: str, file_type: str, csv_index=None, subdir: str = None
    ):
        save_path = self.get_save_path(file_name, file_type, subdir)
        if not op.isfile(save_path):
            raise FileNotFoundError(f"{save_path} not found")
        if file_type == "pkl":
            with open(save_path, "rb") as f:
                data = pickle.load(f)
        elif file_type == "csv":
            data = pd.read_csv(save_path, index_col=csv_index)
        elif file_type == "feather":
            data = pd.read_feather(save_path)
        elif file_type == "pq":
            data = pd.read_parquet(save_path)
        elif file_type == "tsv":
            data = pd.read_csv(save_path, sep="\t", header=0)
        else:
            raise ValueError(f"{file_type} not supported")
        print(f"Successfully loaded cache file from {save_path}")
        return data

    def cache_exists(self, file_name: str, file_type: str, subdir: str = None):
        return op.isfile(self.get_save_path(file_name, file_type, subdir))
