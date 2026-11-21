"""EdgeLayer K1 · KV cache size calculator.

KV bytes = 2 x layers x kv_heads x head_dim x tokens x bytes_per_value (x batch)

Usage:
    python kv_size.py Qwen/Qwen3-0.6B
    python kv_size.py meta-llama/Llama-3.1-8B --tokens 4096 32768 131072 --weights-gb 16.06
    python kv_size.py Qwen/Qwen3-0.6B --batch 8 --bytes 1        # e.g. an 8-bit cache

Gated models (Llama) need `huggingface-cli login` first.
"""
import argparse


def kv_bytes_per_token(config, bytes_per_value=2):
    """Cache bytes for one token of one sequence, from a Hugging Face config."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads  # MHA if missing
    return 2 * config.num_hidden_layers * kv_heads * head_dim * bytes_per_value


def kv_bytes(model_id, tokens, bytes_per_value=2, batch=1):
    """Total KV cache bytes for `batch` sequences of `tokens` tokens (the snippet from the post)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_id)
    return kv_bytes_per_token(c, bytes_per_value) * tokens * batch


def main():
    p = argparse.ArgumentParser(description="KV cache size for any Hugging Face model")
    p.add_argument("model_id")
    p.add_argument("--tokens", type=int, nargs="+", default=[4096, 32768, 131072])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--bytes", type=float, default=2, help="bytes per number: 2 = BF16/FP16, 1 = FP8/INT8")
    p.add_argument("--weights-gb", type=float, default=None,
                   help="model weight size in GB, to print where the cache passes the weights")
    a = p.parse_args()

    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(a.model_id)
    per_tok = kv_bytes_per_token(c, a.bytes)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads

    print(f"{a.model_id}")
    print(f"  layers={c.num_hidden_layers}  kv_heads={c.num_key_value_heads}  head_dim={head_dim}  "
          f"max_context={getattr(c, 'max_position_embeddings', 'n/a')}")
    print(f"  cache per token: {per_tok:,.0f} bytes ({per_tok / 1024:.0f} KiB)  at {a.bytes:g} bytes/number")
    for t in a.tokens:
        print(f"  {t:>8,} tokens x batch {a.batch}: {per_tok * t * a.batch / 1e9:8.2f} GB")
    if a.weights_gb:
        print(f"  cache passes {a.weights_gb:g} GB of weights at ~{a.weights_gb * 1e9 / (per_tok * a.batch):,.0f} tokens")


if __name__ == "__main__":
    main()
