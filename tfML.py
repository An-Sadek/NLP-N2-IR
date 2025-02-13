import os 
import math

import numpy as np
import polars as pl

import tensorflow as tf
from tensorflow import keras
from keras import layers, utils


class SteamDataset(utils.Sequence):

    def __init__(self, path: str, batch_size: int, n_rows: int):
        super().__init__()

        assert os.path.exists(path), "Duong dan khong ton tai"
        self.data = pl.read_csv(path)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.batch_size
        
        curr_data = self.data[start_idx: end_idx, "review_text"]
        curr_label = self.data[start_idx: end_idx, "review_votes"]

        batch_data = np.zeros((len(curr_data),))



class LogisticRegression(layers.Layer):

    def __init__(self):
        pass

    def call(self):
        pass

