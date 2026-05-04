"""
Network Traffic Classification — Results Visualization
=======================================================
Generates 5 publication-quality plots for all 6 models:
  1. Accuracy & F1-Macro comparison (bar chart)
  2. Train / Validation / Test accuracy per model (grouped bars)
  3. Per-class F1-Score heatmap
  4. Confusion matrices (6-panel grid)
  5. Radar / spider chart — multi-metric overview

Run:
    pip install matplotlib seaborn numpy scikit-learn
    python visualize_results.py

Outputs:  results_comparison.png
          train_val_test.png
          perclass_f1_heatmap.png
          confusion_matrices.png
          radar_chart.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from math import pi

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "Logistic Regression":  "#4E79A7",
    "Decision Tree":        "#9B59B6",
    "SVM":                  "#2ECC71",
    "Random Forest":        "#F4B942",
    "Gradient Boosting":    "#E74C3C",
    "Ensemble (SVM+RF+GB)": "#1ABC9C",
}
BG       = "#0F1117"
CARD_BG  = "#1A1D27"
TEXT     = "#E8EAF0"
SUBTEXT  = "#8B90A0"
GRID_CLR = "#2A2D3A"
FONT_TITLE = dict(fontsize=14, fontweight="bold", color=TEXT, fontfamily="monospace")
FONT_SUB   = dict(fontsize=10, color=SUBTEXT, fontfamily="monospace")
FONT_TICK  = dict(fontsize=9,  color=SUBTEXT, fontfamily="monospace")
FONT_VAL   = dict(fontsize=8,  color=TEXT,    fontfamily="monospace", fontweight="bold")

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   SUBTEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "grid.color":        GRID_CLR,
    "text.color":        TEXT,
    "font.family":       "monospace",
})

# ── Data ─────────────────────────────────────────────────────────────────────
MODELS  = ["Logistic\nRegression", "Decision\nTree", "SVM",
           "Random\nForest", "Gradient\nBoosting", "Ensemble\n(SVM+RF+GB)"]
MODELS_SHORT = ["LR", "DT", "SVM", "RF", "GB", "ENS"]
MODEL_KEYS   = ["Logistic Regression", "Decision Tree", "SVM",
                "Random Forest", "Gradient Boosting", "Ensemble (SVM+RF+GB)"]
CLASSES = ["Amazon", "Hulu", "YouTube"]

# Test-set metrics
TEST_ACC = [29.17, 43.06, 41.67, 43.06, 54.17, 50.00]
TEST_F1  = [27.10, 39.90, 37.87, 41.60, 53.14, 48.53]
TEST_PRE = [27.98, 49.89, 45.93, 45.24, 55.17, 49.31]
TEST_REC = [29.76, 44.05, 41.96, 41.27, 51.82, 48.70]

# Train / Val / Test accuracy
TRAIN_ACC = [42.06, 47.20, 55.14, 99.07, 99.07, 99.07]
VAL_ACC   = [43.06, 43.06, 37.50, 38.89, 43.06, 38.89]

# Per-class F1 on test set  [amazon, hulu, youtube]
PERCLASS_F1 = {
    "Logistic Regression":  [34.0, 29.0, 14.0],
    "Decision Tree":        [43.9, 50.0, 25.8],
    "SVM":                  [55.0, 28.0, 33.0],
    "Random Forest":        [51.0, 40.0, 33.0],
    "Gradient Boosting":    [64.0, 46.0, 49.0],
    "Ensemble (SVM+RF+GB)": [60.0, 42.1, 43.5],
}

# Confusion matrices [actual rows × predicted cols] → [amazon, hulu, youtube]
CONF_MATRICES = {
    "Logistic Regression":  np.array([[ 7, 19,  2], [ 3, 15,  4], [ 4, 17,  1]]),
    "Decision Tree":        np.array([[ 9, 16,  3], [ 2, 18,  2], [ 2, 16,  4]]),
    "SVM":                  np.array([[15, 10,  3], [ 6, 14,  2], [ 3, 16,  3]]),
    "Random Forest":        np.array([[15, 11,  2], [ 5, 13,  4], [ 2, 17,  3]]),
    "Gradient Boosting":    np.array([[18,  5,  5], [ 3, 19,  0], [ 2, 14,  6]]),
    "Ensemble (SVM+RF+GB)": np.array([[18,  4,  6], [ 6,  8,  8], [ 8,  4, 10]]),
}

COLORS = [PALETTE[k] for k in MODEL_KEYS]


# ════════════════════════════════════════════════════════════════════════════
# 1 ─ Accuracy & F1 Comparison
# ════════════════════════════════════════════════════════════════════════════
def plot_comparison():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD_BG)

    x = np.arange(len(MODELS))
    w = 0.35

    bars_acc = ax.bar(x - w/2, TEST_ACC, w, color=COLORS, alpha=0.9,
                      edgecolor=BG, linewidth=1.2, label="Test Accuracy")
    bars_f1  = ax.bar(x + w/2, TEST_F1,  w, color=COLORS, alpha=0.55,
                      edgecolor=BG, linewidth=1.2, hatch="//", label="F1-Macro")

    # Value labels
    for bar, val in zip(bars_acc, TEST_ACC):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", **FONT_VAL)
    for bar, val in zip(bars_f1, TEST_F1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", **FONT_VAL)

    # Baseline reference line (Decision Tree F1)
    ax.axhline(39.9, color="#9B59B6", linestyle="--", linewidth=1.3, alpha=0.7)
    ax.text(5.6, 41.2, "DT Baseline F1", color="#9B59B6", fontsize=8, fontfamily="monospace")

    ax.set_xticks(x); ax.set_xticklabels(MODELS, **FONT_TICK)
    ax.set_ylim(0, 72); ax.set_ylabel("Score (%)", **FONT_SUB)
    ax.set_title("Test Accuracy & F1-Macro — All Models", **FONT_TITLE, pad=14)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)

    legend = ax.legend(loc="upper left", framealpha=0.2,
                       facecolor=CARD_BG, edgecolor=GRID_CLR,
                       labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print("Saved: results_comparison.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 2 ─ Train / Val / Test Accuracy
# ════════════════════════════════════════════════════════════════════════════
def plot_train_val_test():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD_BG)

    x = np.arange(len(MODELS)); w = 0.25

    b1 = ax.bar(x - w, TRAIN_ACC, w, color=COLORS, alpha=0.9,
                edgecolor=BG, linewidth=1.2, label="Train Accuracy")
    b2 = ax.bar(x,     VAL_ACC,   w, color=COLORS, alpha=0.55,
                edgecolor=BG, linewidth=1.2, hatch="..", label="Val Accuracy")
    b3 = ax.bar(x + w, TEST_ACC,  w, color=COLORS, alpha=0.75,
                edgecolor=BG, linewidth=1.2, hatch="xx", label="Test Accuracy")

    for bars, vals in [(b1, TRAIN_ACC), (b2, VAL_ACC), (b3, TEST_ACC)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", fontsize=7,
                    color=TEXT, fontfamily="monospace", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(MODELS, **FONT_TICK)
    ax.set_ylim(0, 115); ax.set_ylabel("Accuracy (%)", **FONT_SUB)
    ax.set_title("Train / Validation / Test Accuracy — Overfitting Analysis", **FONT_TITLE, pad=14)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.2, facecolor=CARD_BG,
              edgecolor=GRID_CLR, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    plt.savefig("train_val_test.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("Saved: train_val_test.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 3 ─ Per-Class F1 Heatmap
# ════════════════════════════════════════════════════════════════════════════
def plot_heatmap():
    matrix = np.array([PERCLASS_F1[k] for k in MODEL_KEYS])  # (6, 3)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD_BG)

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=70, aspect="auto")

    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, **FONT_TICK)
    ax.set_yticks(range(6)); ax.set_yticklabels(MODEL_KEYS, fontsize=9,
                                                 color=TEXT, fontfamily="monospace")
    ax.set_title("Per-Class F1-Score Heatmap — Test Set", **FONT_TITLE, pad=14)

    for i in range(6):
        for j in range(3):
            val = matrix[i, j]
            fg = "black" if val > 40 else TEXT
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=fg, fontfamily="monospace")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
    cbar.set_label("F1-Score (%)", color=SUBTEXT, fontsize=9, fontfamily="monospace")

    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig("perclass_f1_heatmap.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("Saved: perclass_f1_heatmap.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 4 ─ Confusion Matrices (3×2 grid)
# ════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Confusion Matrices — Test Set", fontsize=16,
                 fontweight="bold", color=TEXT, fontfamily="monospace", y=1.01)

    for ax, key, title in zip(axes.flat, MODEL_KEYS, MODEL_KEYS):
        cm = CONF_MATRICES[key]
        color = PALETTE[key]

        # Custom colormap per model
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("m", [CARD_BG, color])

        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        ax.set_facecolor(CARD_BG)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=8,
                                                     color=SUBTEXT, fontfamily="monospace")
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES, fontsize=8,
                                                     color=SUBTEXT, fontfamily="monospace")
        ax.set_xlabel("Predicted", fontsize=8, color=SUBTEXT, fontfamily="monospace")
        ax.set_ylabel("Actual",    fontsize=8, color=SUBTEXT, fontfamily="monospace")
        ax.set_title(title, fontsize=10, fontweight="bold", color=color,
                     fontfamily="monospace", pad=8)
        ax.spines[["top","right","left","bottom"]].set_color(GRID_CLR)
        ax.tick_params(length=0)

        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                bright = val > cm.max() * 0.5
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="black" if bright else TEXT,
                        fontfamily="monospace")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("Saved: confusion_matrices.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 5 ─ Radar Chart — Multi-Metric Overview
# ════════════════════════════════════════════════════════════════════════════
def plot_radar():
    metrics      = ["Test Acc", "F1-Macro", "Precision", "Recall", "Val Acc"]
    n            = len(metrics)
    angles       = [pi/2 + 2*pi*i/n for i in range(n)] + [pi/2]  # close loop

    data = {
        "Logistic Regression":  [29.17, 27.10, 27.98, 29.76, 43.06],
        "Decision Tree":        [43.06, 39.90, 49.89, 44.05, 43.06],
        "SVM":                  [41.67, 37.87, 45.93, 41.96, 37.50],
        "Random Forest":        [43.06, 41.60, 45.24, 41.27, 38.89],
        "Gradient Boosting":    [54.17, 53.14, 55.17, 51.82, 43.06],
        "Ensemble (SVM+RF+GB)": [50.00, 48.53, 49.31, 48.70, 38.89],
    }

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor(BG)
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(CARD_BG)
    ax.spines["polar"].set_color(GRID_CLR)

    # Grid rings
    for r in [20, 40, 60, 80]:
        ax.plot(np.linspace(0, 2*pi, 200), [r]*200,
                color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.5)
        ax.text(pi/2, r+2, f"{r}%", ha="center", fontsize=7,
                color=SUBTEXT, fontfamily="monospace")

    # Spokes
    for a, label in zip(angles[:-1], metrics):
        ax.plot([a, a], [0, 85], color=GRID_CLR, linewidth=0.8, alpha=0.5)
        ax.text(a, 92, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=TEXT, fontfamily="monospace")

    # Model polygons
    for key in MODEL_KEYS:
        vals   = data[key] + [data[key][0]]
        color  = PALETTE[key]
        ax.plot(angles, vals, color=color, linewidth=2.2, alpha=0.9)
        ax.fill(angles, vals, color=color, alpha=0.08)
        ax.scatter(angles[:-1], data[key], color=color, s=40, zorder=5)

    ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Multi-Metric Radar — All Models (Test Set)",
                 fontsize=15, fontweight="bold", color=TEXT,
                 fontfamily="monospace", pad=30)

    legend_handles = [
        mpatches.Patch(facecolor=PALETTE[k], label=k, alpha=0.85)
        for k in MODEL_KEYS
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              framealpha=0.2, facecolor=CARD_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    plt.savefig("radar_chart.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("Saved: radar_chart.png")
    plt.close()


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_comparison()
    plot_train_val_test()
    plot_heatmap()
    plot_confusion_matrices()
    plot_radar()
    print("\nAll 5 plots generated successfully.")
