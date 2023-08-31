# GPT Mini Mini

This repository is based on the [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) repository for learning transformer models.

## Training

```bash
python cli/train.py +configs=<config>
# Example
ptyhon cli/train.py +configs=gpt2mm
```

## Installation

```bash
docker-compose up --build -d
docker-compose down
docker exec -it gptmm-dev /bin/bash
```

## TODO

- [ ] schedure learning rate (warmup)
- [ ] pretrainer gpt2 model
- [ ] support to float16
- [ ] improve perplexity metric
- [ ] add tensorboard
- [ ] add wandb
- [ ] add suppprt to distributed training
- [ ] add support to openia tokenizer

## Acknowledgements

[Andrej Karpathy](https://karpathy.ai/) for sharing his knowledge with us all. This repository is based on the [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) repository for learning transformer models.

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master)
- [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main)
- [Pytorch text](https://github.com/pytorch/text/tree/main)
- [MULTIHEADATTENTION](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [SCALED DOT PRODUCT ATTENTION (SDPA)](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
