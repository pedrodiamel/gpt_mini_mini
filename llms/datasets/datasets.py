import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        """CharDataset
        Args:
            data (List): raw data
            block_size (int): block size for context window
        Ref:
            https://github.com/facebookresearch/xformers/blob/main/examples/microGPT.py
        """
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

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
