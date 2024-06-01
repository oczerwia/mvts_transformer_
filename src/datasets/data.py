import glob
import logging
import os
import re
from itertools import chain, repeat, cycle, islice
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import pandas as pd
from datasets import utils
from tqdm import tqdm

logger = logging.getLogger("__main__")


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=None, factor=1):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class CedalionfNIRS(BaseData):
    """Load datasets from the same format as cedalion reads them.

    In the end, the data needs to be in the shape (sample, channel).
    Going furhter, all subjects need to be concatinated on the 0 axis.

    Since any preprocessing steps will be performed in Cedalion, we only transform shapes.


    # FIXME: if we want classification or regression, we need labels (should be checked when using contrastive learning)
    """

    def __init__(
        self,
        root_dir,
        file_list=None,
        pattern=None,
        n_proc=1,
        limit_size=None,
        config=None,
    ):
        # self.set_num_processes(n_proc=n_proc)

        self.config = config
        self.set_num_processes(n_proc=n_proc)

        self.root_dir = root_dir
        self.file_list = file_list
        self.pattern = pattern

        self.all_df = self.load_all(
             root_dir, file_list=file_list, pattern=pattern
        )
        self.all_IDs = ((list(range(len(self.all_df))))) # monkey patch


        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        
        # use all features
        self.feature_names = self.sample_single_file(self.all_df[0]).columns
        # Normally here to exclued features, we don't do that here
        self.feature_df = self.all_df 

    def sample_single_file(self, file_path):
        """is used to look into the file structure."""
        return pd.read_csv(file_path)

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, "*"))  # list of all paths

        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".csv")
        ]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))


        return input_paths # ONLY RETURNS LIST OF RELEVANT PATHS


    def __len__(self):
        return len(self.load_all(self.root_dir,
                                 self.file_list,
                                 self.pattern))


data_factory = {
    "fnirs": CedalionfNIRS,
}

