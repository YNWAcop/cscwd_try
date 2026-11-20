from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from .paths import RESULTS, OUTPUT


def framework(dest):
    fig, ax = plt.subplots(figsize=(8.5, 1.95))
    ax.set(xlim=(0, 100), ylim=(0, 22))
    ax.axis("off")
    font = FontProperties(size=10)
    data = [
        (
            0,
            "Protect scores",
            ["Local and contextual readings", "Intersect label-wise intervals"],
            "#edf4e6",
        ),
        (
            25,
            "Find feasible lists",
            ["Shared threshold and budget", "Minimum movement per swap"],
            "#edf3fc",
        ),
        (
            50,
            "Verify candidates",
            ["Read only feasible endpoints", "Batch or adapt across requests"],
            "#fff4dc",
        ),
        (
            75,
            "Show evidence",
            ["Choose the best feasible list", "Audit evidence gained and lost"],
            "#f7ecf0",
        ),
    ]
    for i, (x, title, body, color) in enumerate(data):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.3, 2.5),
                22.9,
                17.2,
                boxstyle="round,pad=.13,rounding_size=.8",
                ec="#7b8587",
                fc=color,
                lw=0.7,
            )
        )
        ax.text(x + 1.4, 16.2, title, fontproperties=font)
        for j, line in enumerate(body):
            ax.text(x + 1.5, 12.2 - 2.7 * j, line, fontsize=7.2)
        if i < 3:
            ax.annotate("", (x + 25.25, 11), (x + 23.2, 11), arrowprops=dict(arrowstyle="->", lw=1))
    ax.plot([3, 18], [5.8, 5.8], c="#8493a0", lw=0.7)
    ax.plot([11, 17], [5.8, 5.8], c="#559777", lw=3)
    for x, y in [(29, 5), (29, 7), (38, 5), (38, 7)]:
        ax.scatter(x, y, s=20, c="#67a19b")
    ax.plot([29, 38], [5, 7], c="#5a7884", lw=0.8)
    ax.plot([29, 38], [7, 5], c="#5a7884", lw=0.8)
    for x in (54, 57, 60, 63, 66):
        ax.scatter(x, 5.5, s=22, c="#83a8c2" if x < 63 else "#ddd")
    for j in range(3):
        ax.text(
            78 + j * 5,
            5.0,
            str(j + 1),
            ha="center",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="#fff", ec="#969da0", lw=0.5),
        )
    fig.tight_layout(pad=0.04)
    fig.savefig(dest / "framework.pdf", bbox_inches="tight")
    plt.close(fig)


def evidence(dest, arms):
    colors = ["#965B3D", "#32766F"]
    rows = [arms[key] for key in ("budget+lock", "floor+budget+lock")]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.15))
    for ax, label, vals in zip(
        axes,
        ("A   Net labeled hits", "B   Queries below floor", "C   Verifier scores"),
        (
            [r["add"] - r["lost"] for r in rows],
            [r["bad_queries"] for r in rows],
            [r["probes"] for r in rows],
        ),
    ):
        ax.barh([1, 0], vals, height=0.43, color=colors)
        ax.set(
            yticks=[1, 0],
            yticklabels=["Budget + lock", "With floor"] if ax is axes[0] else [],
            ylim=(-0.5, 1.5),
            xlim=(0, max(vals) * 1.38),
        )
        for y, num in zip((1, 0), vals):
            ax.text(num + max(vals) * 0.045, y, f"{num:,}", va="center", fontsize=10)
        ax.set_title(label, loc="left", size=10, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_xticks([0, 5, 10, 15, 20])
    axes[1].set_xticks([0, 50, 100, 150])
    axes[2].set_xticks([0, 500, 1000, 1500, 2000])
    fig.tight_layout(w_pad=1)
    fig.savefig(dest / "evidence.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dest = OUTPUT / "figures"
    dest.mkdir(exist_ok=True)
    framework(dest)
    evidence(dest, json.loads((RESULTS / "mechanisms.json").read_text())["arms"])
