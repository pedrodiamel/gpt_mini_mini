import os

import pytest
from llms.datasets.datasets import CharDataset

# Input parameters
PATH_DATASET = "/.datasets/llms/brasiliansong/input.txt"
BLOCK = 16


def test_tiny_dataset():
    dataset = CharDataset(PATH_DATASET, BLOCK, train=True, download=True)
    assert len(dataset) > 0

    x, y = dataset[0]
    print(dataset.voc)

    print("")
    print("len: ", len(dataset))
    print("x: ", x)
    print("decode:", dataset.from_tokens(x))
    print("y: ", y)
    print("decode:", dataset.from_tokens(y))


if __name__ == "__main__":
    pytest.main([__file__])
