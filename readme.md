# LoRA from Scratch — PyTorch

Implementation of [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) using PyTorch's native parametrization API.

## Approach

We train a simple 3-layer MLP (`BaseModel`) on all MNIST digits, then fine-tune it on a single digit using LoRA — without touching the original weights.

For each linear layer with weight matrix `W (d×k)`, we introduce two low-rank matrices:

```
W_new = W + scale * B @ A
```

Where `A (r×k)` and `B (d×r)` are the only trainable parameters. With `r=1`, this is **< 0.25%** of the original parameter count.

PyTorch's `parametrize.register_parametrization` hooks into weight access transparently — the model computes `W + B@A` on every forward pass without any changes to the model architecture.

## Why this works

The key insight is **matrix rank** — weight update matrices during fine-tuning are inherently low-rank, meaning the adaptation needed for a specific task lives in a small subspace. Full details in the article below.

📝 Code reference : [code reference](https://github.com/hkproj/pytorch-lora/blob/main/lora.ipynb)

📝 Medium: `[link](https://medium.com/p/93bbe1c50dfd?postPublishedType=initial)`

📄 Paper: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
