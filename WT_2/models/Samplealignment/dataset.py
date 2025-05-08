import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    """质谱相似性数据集"""

    def __init__(self, pkl_file):
        super().__init__()

        with open(pkl_file, 'rb') as f:
            self.pkl_file = pickle.load(f)

        self.lefts, self.rights, self.left_rts, self.right_rts, self.labels = self.pkl_file['left_intensities'], \
                                                                              self.pkl_file['right_intensities'], \
                                                                              self.pkl_file['left_rt'], \
                                                                              self.pkl_file['right_rt'], \
                                                                              self.pkl_file['labels']

    def __len__(self):
        return self.lefts.shape[0]

    def __getitem__(self, idx):

        return (
            np.squeeze(torch.tensor(self.lefts[idx].toarray().astype(np.float32))),
            np.squeeze(torch.tensor(self.rights[idx].toarray().astype(np.float32))),
            torch.tensor(self.left_rts[idx]),
            torch.tensor(self.right_rts[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


