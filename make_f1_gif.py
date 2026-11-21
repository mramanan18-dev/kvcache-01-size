"""EdgeLayer K1 · F1 as an animated GIF: one decoding step.

Order (as in a real decode step):
  the latest input token ('the') makes Q, K, V -> its K, V are appended to the cache ->
  Q is scored against every stored key (its own included) -> softmax weights ->
  weighted mix of values -> output predicts the next token ('mat').
"""
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from make_figures import box, arrow, OUT, INK, BLUE, ORANGE, GREEN, GREY

WORDS = ["The", "cat", "sat", "on", "the"]          # last one = latest input token
WEIGHTS = [0.05, 0.40, 0.15, 0.10, 0.30]
X0, W, GAP = 1.7, 1.2, 0.3
CENTERS = [X0 + i * (W + GAP) + W / 2 for i in range(5)]
MID = (CENTERS[0] + CENTERS[4]) / 2
STEP_TEXT = {
    0: "The KV cache so far: one key and one value for each earlier token",
    1: "1. The latest token, 'the', makes a query, a key and a value. Its key and value join the cache",
    2: "2. Score its query against every stored key, its own included",
    3: "3. Turn the scores into weights that add up to 1",
    4: "4. Mix the stored values, weighted. The output predicts the next token: 'mat'",
}


def draw(step, n_arrows=5):
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=100)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(0.25, 5.1, STEP_TEXT[step], fontsize=12, color=INK, fontweight="semibold", va="center")

    n = 4 if step == 0 else 5
    right = X0 + n * W + (n - 1) * GAP
    ax.add_patch(FancyBboxPatch((X0 - 0.12, 1.92), right - X0 + 0.24, 1.66,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="none", ec=GREY, lw=1, ls="--"))
    ax.text(right + 0.25, 2.75, f"KV cache:\n{n} tokens", va="center", fontsize=10, color="#555")
    ax.text(X0 - 0.2, 3.17, "keys", ha="right", va="center", fontsize=10, color=BLUE)
    ax.text(X0 - 0.2, 2.32, "values", ha="right", va="center", fontsize=10, color=ORANGE)

    for i in range(n):
        c, wd = CENTERS[i], WORDS[i]
        new = i == 4
        box(ax, c - W / 2, 2.9, W, 0.55, f"K: {wd}", fc="#dbe7f5", ec=GREEN if new else BLUE, fs=10,
            weight="semibold" if new else "normal")
        box(ax, c - W / 2, 2.05, W, 0.55, f"V: {wd}", fc="#fde9d4", ec=GREEN if new else ORANGE, fs=10,
            weight="semibold" if new else "normal")

    if step == 0:
        box(ax, CENTERS[4] - W / 2, 4.3, W, 0.55, "input: 'the'", fc="#dff3e7", ec=GREEN, fs=10)
        ax.text(CENTERS[4], 3.95, "next in", ha="center", fontsize=9.5, color=GREEN)

    if step >= 1:
        box(ax, MID - 1.3, 4.3, 2.6, 0.55, "Q from latest token 'the'", fc="#dff3e7", ec=GREEN, fs=10.5,
            weight="semibold")
    if step == 1:
        ax.text(CENTERS[4], 3.65, "new", ha="center", fontsize=10, color=GREEN, fontweight="semibold")

    if step >= 2:
        for i in range(min(n_arrows, 5)):
            c, a = CENTERS[i], WEIGHTS[i]
            lw = 1.0 if step == 2 else 0.6 + 5 * a
            arrow(ax, MID, 4.28, c, 3.47, color=GREEN, lw=lw, alpha=0.85)
            if step >= 3:
                ax.text(c - 0.45 if c < MID - 0.1 else c + 0.22, 3.62, f"{a:.2f}",
                        fontsize=10, color=INK, ha="center")

    if step == 4:
        for i in range(5):
            c, a = CENTERS[i], WEIGHTS[i]
            arrow(ax, c, 2.03, MID, 1.07, color=ORANGE, lw=0.6 + 5 * a, alpha=0.85)
        box(ax, MID - 1.5, 0.5, 3.0, 0.55, "output  →  next token: 'mat'", fc="#fff", ec=ORANGE, fs=10.5)
        ax.text(MID, 0.15, "Next step: 'mat' goes in, adds its own key and value, and the cache grows to 6",
                ha="center", fontsize=10, color="#555", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    frames, durations = [], []

    def add(img, ms):
        frames.append(img)
        durations.append(ms)

    add(draw(0), 1600)
    add(draw(1), 2400)
    for k in range(1, 6):                 # score arrows appear one by one
        add(draw(2, n_arrows=k), 350)
    add(draw(2), 800)
    add(draw(3), 1800)
    add(draw(4), 3000)

    pal = frames[-1].quantize(colors=128, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    path = os.path.join(OUT, "f1_decode_step.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)

    draw(4).save(os.path.join(OUT, "f1_decode_step_still.png"))
    print(path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main()
