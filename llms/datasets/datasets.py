import os

import torch
from torch.utils.data import Dataset

from . import utils


class CharDataset(Dataset):
    # this for the tinyshakespeare dataset
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    MD5 = "6fb458f1232090904fb40fe944165e91"

    def __init__(self, pathname, block_size, train=True, download=False):
        """CharDataset
        Args:
            pathname (str): path to dataset
            block_size (int): block size for context window
            train (bool): train or test
            download (bool): download dataset if not found
        Ref:
            https://github.com/facebookresearch/xformers/blob/main/examples/microGPT.py
        """

        # if download:
        #     utils.download_url(self.URL, pathname, self.MD5, False)

        if not os.path.isfile(pathname):
            raise FileNotFoundError("Dataset not found.")

        with open(pathname, "r", encoding="utf-8") as f:
            data = f.read()

        voc = sorted(list(set(data)))
        vocab_size = len(voc)

        self.stoi = {ch: i for i, ch in enumerate(voc)}
        self.itos = {i: ch for i, ch in enumerate(voc)}

        n = int(0.9 * len(data))
        data = data[:n] if train else data[n:]
        data_size = len(data)

        self.pathname = pathname
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.count = data_size
        self.voc = voc

    def __len__(self):
        return self.count - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def to_tokens(self, message, device):
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[None, ...].to(device)

    def from_tokens(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])
