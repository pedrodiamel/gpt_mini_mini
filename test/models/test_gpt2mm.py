import torch
from llms.models.gpt2mm import gpt2mm
from torch.autograd import Variable


def test_gpt2mm():
    batch_size = 1
    block_size = 8
    n_layers = 6
    vocab_size = 27  # get to dataset
    n_embd = 32
    n_heads = 4

    model = gpt2mm(
        False, n_layers=n_layers, vocab_size=vocab_size, n_embd=n_embd, n_heads=n_heads, block_size=block_size
    )

    x = Variable(torch.randint(0, vocab_size, (batch_size, block_size)))
    y = model(x)
    print(y.size())
    print(y[0, -1, :])
