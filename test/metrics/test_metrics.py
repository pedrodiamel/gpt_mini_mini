import torch
from torcheval.metrics.text import Perplexity


def test_perplexity():
    preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
    target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))

    print(preds.shape)
    print(target.shape)

    metric = Perplexity()

    metric.update(preds, target)
    perp = metric.compute()
    print(perp)
