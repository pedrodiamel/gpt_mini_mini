# GPT Mini Mini

## Training

```bash
cd runs
python helper/train.py +configs=<config>
```

## Installation

```bash
docker-compose up --build -d
docker-compose down
docker exec -it gptmm-dev /bin/bash
```

## TODO

- [ ] schedure learning rate
- [ ] pretrainer gpt2
- [ ] suport toqueniezer

## References

- [MULTIHEADATTENTION](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [SCALED DOT PRODUCT ATTENTION (SDPA)](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
