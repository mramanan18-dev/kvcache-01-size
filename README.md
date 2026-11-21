# edgelayer-k1-kv-size

Code for EdgeLayer's KV-cache series, post 1: *The memory that grows: why long context runs out of GPU memory*.

| File | What it does |
|---|---|
| `kv_size.py` | KV cache calculator for any Hugging Face model (`kv_bytes()` from the post + a small CLI) |
| `make_figures.py` | F1 (static), F2 memory vs context, F3 speed ceiling, F4 head sharing, F5 series map, equation images |
| `make_f1_gif.py` | F1 as an animated GIF + a still version for email |
| `make_tables.py` | T1 and T2 as PNGs |

## Run

```bash
pip install -r requirements.txt

python kv_size.py Qwen/Qwen3-0.6B --weights-gb 1.19
python kv_size.py Qwen/Qwen3-0.6B --batch 8 --bytes 1     # 8 sequences, 8-bit cache

python make_figures.py      # writes ./figures
python make_f1_gif.py
python make_tables.py
```

Gated models like Llama need `huggingface-cli login` before `kv_size.py` can read their config.

## Notes

- All numbers are analytic: `KV bytes = 2 × layers × KV heads × head dim × tokens × bytes per number`, one sequence, no framework overhead.
- Speed ceilings use vendor peak bandwidths (T4 320 GB/s, H100 SXM 3.35 TB/s) and assume every byte of weights and cache is read once per token.
- Measured numbers come in the benchmark post at the end of the series.

License: code MIT · figures and tables CC BY 4.0
