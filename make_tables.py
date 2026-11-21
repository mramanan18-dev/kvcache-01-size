"""EdgeLayer K1 · table PNGs (T1 cache size, T2 levers)."""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from make_figures import OUT, INK

HEAD_BG = "#1f2933"
HEAD_FG = "#ffffff"
ZEBRA = "#f4f6f9"
LINE = "#d9dee5"
HILITE = "#e8f1fb"


def render(name, header, rows, col_w, wrap, bold_rows=(), hilite_rows=(), fs=10.5):
    """col_w: column widths in inches; wrap: max chars per line per column."""
    line_h = 0.21 * fs / 10.5
    pad_x, pad_y = 0.10, 0.10

    def cells(row):
        return [textwrap.wrap(str(t), w) or [""] for t, w in zip(row, wrap)]

    all_rows = [header] + rows
    wrapped = [cells(r) for r in all_rows]
    heights = [max(len(c) for c in r) * line_h + 2 * pad_y for r in wrapped]
    W, H = sum(col_w), sum(heights)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    y = 0
    for ri, (r, h) in enumerate(zip(wrapped, heights)):
        is_head = ri == 0
        data_i = ri - 1
        if is_head:
            bg = HEAD_BG
        elif data_i in hilite_rows:
            bg = HILITE
        else:
            bg = ZEBRA if data_i % 2 else "#ffffff"
        ax.add_patch(Rectangle((0, y), W, h, fc=bg, ec="none"))
        x = 0
        for ci, (lines, cw) in enumerate(zip(r, col_w)):
            weight = "semibold" if (is_head or ci == 0 or data_i in bold_rows) else "normal"
            ax.text(x + pad_x, y + pad_y, "\n".join(lines), va="top", ha="left",
                    fontsize=fs, color=HEAD_FG if is_head else INK, fontweight=weight,
                    linespacing=1.35)
            x += cw
        ax.plot([0, W], [y + h, y + h], color=LINE, lw=0.8)
        y += h

    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white")
    plt.close(fig)
    print("saved", name)


def t1():
    header = ["Model", "Layers × KV heads × head size", "Cache per token", "At 4k", "At 32k",
              "At 128k", "Weights (BF16)", "Cache passes the weights at"]
    rows = [
        ["Qwen3-0.6B", "28 × 8 × 128", "112 KiB", "0.47 GB", "3.8 GB", "beyond its context", "1.19 GB", "~10k tokens"],
        ["Llama 3.1 8B", "32 × 8 × 128", "128 KiB", "0.54 GB", "4.3 GB", "17.2 GB", "16.1 GB", "~123k tokens"],
        ["Llama 3.1 70B", "80 × 8 × 128", "320 KiB", "1.3 GB", "10.7 GB", "43.0 GB", "141 GB", "~431k tokens"],
    ]
    render("t1_cache_size.png", header, rows,
           col_w=[1.35, 1.45, 1.0, 0.8, 0.8, 1.05, 1.05, 1.35],
           wrap=[14, 15, 9, 8, 8, 10, 9, 14])


def t2():
    header = ["Approach", "How much smaller", "Quality risk", "Needs a new model?", "Where you'll find it", "Post"]
    rows = [
        ["8-bit cache (FP8, q8_0)", "About 2×", "Low", "No", "vLLM, TensorRT-LLM, llama.cpp, Ollama", "Next"],
        ["K/V-aware 2 to 4 bit (KIVI, KVQuant)", "4 to 8× fewer bits", "Medium", "No", "Research code, parts in Hugging Face", "3"],
        ["Rotate, then round (TurboQuant)", "About 4.5× fewer bits", "Debated", "No", "Early runtime support", "4"],
        ["NVFP4 cache", "About 2× smaller than FP8", "Under 1% loss reported", "No", "Blackwell GPUs", "5"],
        ["Transform coding (KVTC)", "Up to 20×", "Low in the paper", "No, brief calibration", "Research; aimed at storing caches between turns", "5"],
        ["Eviction", "Whatever budget you set", "Can drop facts the model needs later", "No", "Some runtimes", "5"],
        ["GQA / MLA", "8× on Llama 3.1 70B / 93% on DeepSeek-V2", "None by design", "Yes", "Pick a model that has it", "1 (intro), 5"],
    ]
    render("t2_levers.png", header, rows,
           col_w=[2.0, 1.75, 1.6, 1.25, 2.3, 0.85],
           wrap=[22, 18, 17, 12, 25, 9])


if __name__ == "__main__":
    t1()
    t2()
