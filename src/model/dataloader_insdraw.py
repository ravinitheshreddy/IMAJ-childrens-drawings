# Code structure courtesy of Ludovica (https://github.com/ludovicaschaerf/Cini_TDA)

import sys
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "./src/utils/")

from utils import read_and_preprocess_image, show_images


class InsdrawDataset(Dataset):
    """Insdraw dataset loader."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transformations: callable = None,
    ):
        """
        Parameters
        ----------
        csv_file : str
            Path to the csv file with annotations. Path to train.csv or test.csv
        root_dir : str
            Path to the data directory
        transformations : callable, optional
            Optional transform to be applied
                on a sample.
        resolution : int, optional
            The resolution to resize the image, by default 480
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transformations = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        A = read_and_preprocess_image(
            self.root_dir + self.data.loc[idx, "A_path"],
            self.transformations,
        )
        B = read_and_preprocess_image(
            self.root_dir + self.data.loc[idx, "B_path"],
            self.transformations,
        )
        C = read_and_preprocess_image(
            self.root_dir + self.data.loc[idx, "C_path"],
            self.transformations,
        )

        sample = [A, B, C]

        return sample

    def __show_images__(self, idx):
        show_images(
            [
                self.data.loc[idx, "A_path"],
                self.data.loc[idx, "B_path"],
                self.data.loc[idx, "C_path"],
            ]
        )

    def __reload__(self, csv_file):
        # del self.data
        self.data = pd.read_csv(csv_file)
        print("reloaded data", self.data.shape)
