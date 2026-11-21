"""EdgeLayer K1 · The memory that grows: figures and equations.

All numbers are analytic: KV bytes = 2 x layers x kv_heads x head_dim x tokens x bytes.
Speed ceiling: tok/s <= bandwidth / (weight bytes + KV bytes).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.labelcolor": "#333",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.edgecolor": "#888",
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

INK = "#1f2933"
BLUE = "#2f6db5"      # cache
ORANGE = "#d9822b"    # weights
GREEN = "#2e8b57"
GREY = "#9aa5b1"
LIGHT = "#eef2f7"

# ---------------------------------------------------------------- models
MODELS = {
    # name: (layers, kv_heads, head_dim, weight_bytes_bf16)
    "Qwen3-0.6B": (28, 8, 128, 1.19e9),
    "Qwen3-1.7B": (28, 8, 128, 3.44e9),
    "Llama 3.1 8B": (32, 8, 128, 16.06e9),
    "Llama 3.1 70B": (80, 8, 128, 141.2e9),
}


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def table():
    print(f"{'model':14s} {'KiB/tok':>8s} {'4k GB':>7s} {'32k GB':>7s} {'128k GB':>8s} {'weights':>8s} {'crossover':>10s}")
    for name, (L, h, d, w) in MODELS.items():
        t = kv_bytes_per_token(L, h, d)
        print(f"{name:14s} {t/1024:8.0f} {t*4096/1e9:7.2f} {t*32768/1e9:7.2f} {t*131072/1e9:8.2f} {w/1e9:8.2f} {w/t:10.0f}")


# ---------------------------------------------------------------- F2
def fig_memory():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("Qwen3-0.6B", 32768, "Qwen3-0.6B (native context 32k)"),
              ("Llama 3.1 8B", 131072, "Llama 3.1 8B (context 128k)")]
    for ax, (name, maxctx, title) in zip(axes, panels):
        L, h, d, w = MODELS[name]
        t = kv_bytes_per_token(L, h, d)
        x = np.linspace(0, maxctx, 400)
        ax.plot(x / 1024, t * x / 1e9, color=BLUE, lw=2.4, label="KV cache (BF16)")
        ax.axhline(w / 1e9, color=ORANGE, lw=2, ls="--", label="Weights (BF16)")
        cx = w / t
        ax.plot(cx / 1024, w / 1e9, "o", color=INK, ms=7, zorder=5)
        ax.annotate(f"cache passes the weights\nat ~{cx/1000:.0f}k tokens",
                    xy=(cx / 1024, w / 1e9), xytext=(14, -38) if name.startswith("Qwen") else (-6, -125),
                    textcoords="offset points", ha="left" if name.startswith("Qwen") else "right", fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GREY))
        end = t * maxctx / 1e9
        ax.annotate(f"{end:.1f} GB", xy=(maxctx / 1024, end), xytext=(-4, 6),
                    textcoords="offset points", ha="right", color=BLUE, fontsize=10, fontweight="semibold")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Context length (thousand tokens)")
        ax.set_ylabel("Memory (GB)")
        ax.set_xlim(0, maxctx / 1024 * 1.02)
        ax.set_ylim(0, max(end, w / 1e9) * 1.22)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle("One sequence, batch size 1. The cache grows with every token; the weights don't.",
                 x=0.01, ha="left", fontsize=11, color="#555", y=1.02)
    fig.savefig(os.path.join(OUT, "f2_memory_vs_context.png"))
    plt.close(fig)


# ---------------------------------------------------------------- F3
def fig_speed():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("Qwen3-0.6B", 32768, 320e9, "Qwen3-0.6B on a T4 (320 GB/s)"),
              ("Llama 3.1 8B", 131072, 3.35e12, "Llama 3.1 8B on an H100 SXM (3.35 TB/s)")]
    for ax, (name, maxctx, bw, title) in zip(axes, panels):
        L, h, d, w = MODELS[name]
        t = kv_bytes_per_token(L, h, d)
        x = np.linspace(0, maxctx, 400)
        y = bw / (w + t * x)
        ax.plot(x / 1024, y, color=GREEN, lw=2.4)
        ax.fill_between(x / 1024, 0, y, color=GREEN, alpha=0.08)
        y0, y1 = y[0], y[-1]
        ax.annotate(f"{y0:.0f} tok/s", xy=(0, y0), xytext=(6, 4), textcoords="offset points",
                    color=GREEN, fontweight="semibold", fontsize=10)
        ax.annotate(f"{y1:.0f} tok/s\n({y0/y1:.1f}x slower)", xy=(maxctx / 1024, y1), xytext=(-4, 14),
                    textcoords="offset points", ha="right", color=GREEN, fontweight="semibold", fontsize=10)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Context length (thousand tokens)")
        ax.set_ylabel("Decode ceiling (tokens/sec)")
        ax.set_xlim(0, maxctx / 1024 * 1.02)
        ax.set_ylim(0, y0 * 1.18)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    fig.suptitle("Ceiling = bandwidth / (weights + cache). Real speeds land below these lines.",
                 x=0.01, ha="left", fontsize=11, color="#555", y=1.02)
    fig.savefig(os.path.join(OUT, "f3_speed_ceiling.png"))
    plt.close(fig)


# ---------------------------------------------------------------- helpers for diagrams
def box(ax, x, y, w, h, text="", fc=LIGHT, ec=GREY, color=INK, fs=10, weight="normal", r=0.04):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight=weight)


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.2, alpha=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
                                 color=color, lw=lw, alpha=alpha))


# ---------------------------------------------------------------- F1
def fig_decode_step():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    words = ["The", "cat", "sat", "on", "the"]
    weights = [0.05, 0.40, 0.15, 0.10, 0.30]
    x0, w, gap = 1.6, 1.3, 0.3
    centers = [x0 + i * (w + gap) + w / 2 for i in range(5)]
    mid = (centers[0] + centers[-1]) / 2
    box(ax, mid - 1.2, 4.45, 2.4, 0.55, "Q from latest token 'the'", fc="#dff3e7", ec=GREEN, fs=10, weight="semibold")
    for c, wd, a in zip(centers, words, weights):
        arrow(ax, mid, 4.43, c, 3.47, color=GREEN, lw=0.6 + 5 * a, alpha=0.8)
        box(ax, c - w / 2, 2.9, w, 0.55, f"K: {wd}", fc="#dbe7f5", ec=BLUE, fs=10)
        ax.text(c - 0.45 if c < mid - 0.1 else c + 0.2, 3.62, f"{a:.2f}", fontsize=9.5, color=INK, ha="center")
        box(ax, c - w / 2, 2.05, w, 0.55, f"V: {wd}", fc="#fde9d4", ec=ORANGE, fs=10)
        arrow(ax, c, 2.03, mid, 1.07, color=ORANGE, lw=0.6 + 5 * a, alpha=0.8)
    box(ax, mid - 1.3, 0.5, 2.6, 0.55, "output → next token: 'mat'", fc="#fff", ec=ORANGE, fs=10)
    ax.text(x0 - 0.15, 3.17, "stored\nkeys", ha="right", va="center", fontsize=9.5, color=BLUE)
    ax.text(x0 - 0.15, 2.32, "stored\nvalues", ha="right", va="center", fontsize=9.5, color=ORANGE)
    ax.text(x0 - 0.15, 3.7, "weight", ha="right", va="center", fontsize=9.5, color="#555")
    ax.text(x0 - 0.15, 4.72, "new", ha="right", va="center", fontsize=9.5, color=GREEN)
    ax.add_patch(FancyBboxPatch((x0 - 0.12, 1.92), 5 * w + 4 * gap + 0.24, 1.66,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="none", ec=GREY, lw=1, ls="--"))
    ax.text(x0 + 5 * w + 4 * gap + 0.25, 2.75, "the KV cache:\none key and one\nvalue per token,\nin every layer",
            va="center", fontsize=9.5, color="#555")
    ax.text(mid, 0.12, "1. score the query against every stored key   2. turn scores into weights   "
            "3. mix the stored values", ha="center", fontsize=9.5, color="#555")
    fig.savefig(os.path.join(OUT, "f1_decode_step.png"))
    plt.close(fig)


# ---------------------------------------------------------------- F4
def fig_heads():
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    cols = [("MHA", 8, "8 K/V heads"), ("GQA", 2, "2 K/V heads (groups of 4)"),
            ("MQA", 1, "1 K/V head"), ("MLA", 0, "1 small latent per token")]
    for c, (name, nkv, sub) in enumerate(cols):
        cx = 0.3 + c * 3.0
        ax.text(cx + 1.25, 4.1, name, ha="center", fontsize=12, fontweight="semibold", color=INK)
        ax.text(cx + 1.25, 0.15, sub, ha="center", fontsize=9.5, color="#555")
        qx = [cx + i * 0.32 for i in range(8)]
        for x in qx:
            box(ax, x, 3.2, 0.24, 0.45, fc="#dff3e7", ec=GREEN, r=0.03)
        ax.text(cx - 0.12, 3.42, "Q", ha="right", fontsize=9.5, color=GREEN)
        if nkv > 0:
            span = 8 // nkv
            for k in range(nkv):
                group = qx[k * span:(k + 1) * span]
                mid = (group[0] + group[-1]) / 2 + 0.12
                kw = 0.24
                kx = mid - kw / 2
                box(ax, kx, 1.25, kw, 0.45, fc="#dbe7f5", ec=BLUE, r=0.03)
                box(ax, kx, 0.65, kw, 0.45, fc="#fde9d4", ec=ORANGE, r=0.03)
                for x in group:
                    ax.plot([x + 0.12, mid], [3.18, 1.72], color=GREY, lw=0.8)
            ax.text(cx - 0.12, 1.47, "K", ha="right", fontsize=9.5, color=BLUE)
            ax.text(cx - 0.12, 0.87, "V", ha="right", fontsize=9.5, color=ORANGE)
        else:
            mid = qx[3] + 0.28
            box(ax, mid - 0.45, 1.25, 0.9, 0.45, "latent", fc="#ece3f7", ec="#7b4fb0", fs=9, r=0.03)
            ax.text(mid, 0.87, "K, V rebuilt\nwhen read", ha="center", fontsize=9, color="#7b4fb0")
            for x in qx:
                ax.plot([x + 0.12, mid], [3.18, 1.72], color=GREY, lw=0.8)
    fig.savefig(os.path.join(OUT, "f4_head_sharing.png"))
    plt.close(fig)


# ---------------------------------------------------------------- F5
def fig_map():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    box(ax, 0.3, 1.95, 2.4, 0.9, "The KV cache\ngrows with context", fc="#dbe7f5", ec=BLUE, fs=11, weight="semibold")
    levers = [
        ("Why the cache grows, and GQA / MLA basics", "Post 1", 4.05, True),
        ("Lower precision: FP8, q8_0", "Post 2", 3.25, False),
        ("Keys vs values: KIVI, KVQuant", "Post 3", 2.45, False),
        ("Rotate, then round: TurboQuant", "Post 4", 1.65, False),
        ("Beyond rounding: KVTC, NVFP4, eviction, MLA", "Post 5", 0.85, False),
    ]
    for text, post, y, this in levers:
        box(ax, 3.6, y - 0.3, 4.6, 0.6, text, fc="#fff", ec=BLUE if this else GREY, fs=10.2)
        box(ax, 8.45, y - 0.3, 1.1, 0.6, post, fc="#dbe7f5" if this else "#dff3e7", ec=BLUE if this else GREEN,
            fs=10.5, weight="semibold")
        arrow(ax, 2.72, 2.4, 3.58, y, color=GREY)
    ax.text(9.0, 4.42, "this post", ha="center", fontsize=9, color=BLUE)
    box(ax, 9.8, 0.55, 1.0, 3.8, "Post 6\nmeasure\nwhat you\ncan run", fc="#fde9d4", ec=ORANGE, fs=10.5, weight="semibold")
    ax.text(5.9, 4.62, "Where this series goes", ha="center", fontsize=12, fontweight="semibold", color=INK)
    ax.text(5.9, 0.15, "Quantization and eviction work on the model you already have. GQA and MLA are fixed when a model is trained.",
            ha="center", fontsize=9.5, color="#555")
    fig.savefig(os.path.join(OUT, "f5_series_map.png"))
    plt.close(fig)


# ---------------------------------------------------------------- equations
def equation(name, tex, w=8, h=0.9):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=20, color=INK,
             math_fontfamily="cm")
    fig.savefig(os.path.join(OUT, name), transparent=False, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    table()
    fig_decode_step()
    fig_memory()
    fig_speed()
    fig_heads()
    fig_map()
    equation("e1_kv_size.png",
             r"$\mathrm{KV\ bytes} = 2 \times L \times H_{kv} \times d_{head} \times T \times b$")
    equation("e2_speed_ceiling.png",
             r"$\mathrm{tokens/sec} \;\leq\; \dfrac{B}{S_{weights} + S_{KV}(T)}$", h=1.2)
    print("saved to", OUT)
