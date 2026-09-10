"""
==================================================================
BLS 层数 × 路数 热力图分析 (AUC / GAUC 双图)
==================================================================
输入: 各 (层数, 路数) 组合的 AUC/GAUC
输出: 两张热力图 (PDF 矢量) + 最优区域 + 边际分析

数据格式 (JSON):
{
  "layers": [1, 2, 3, 4, 5],
  "ways":   ["2-way", "3-way", "4-way", "5-way"],
  "auc":    [[...], [...], ...],    # shape [len(layers), len(ways)]
  "gauc":   [[...], [...], ...]     # 可选
}

用法:
  python bls_cross_analysis.py --results results_grid.json --out_dir /root/autodl-fs/log-总/analysis/output
==================================================================
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_single_heatmap(ax, mat, layers, ways, title, cmap, fmt=".5f"):
    """在给定 ax 上画一个美观的热力图。
    风格: 细白网格线 + 最大值加粗黑框 + 黑底白字。"""
    mat = np.array(mat)
    n_rows, n_cols = mat.shape
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    # ── 细白网格线 (参考样式) ──
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", size=0)

    ax.set_xticklabels(ways, fontsize=10)
    ax.set_yticklabels([f"L={l}" for l in layers], fontsize=10)
    ax.set_xlabel("Cross Ways", fontsize=11)
    ax.set_ylabel("BLS Layers", fontsize=11)
    ax.set_title(title, fontsize=12)

    # ── 最大值: 加粗黑框 (不填充) + 白字 (最大值格本身色深) ──
    best = np.unravel_index(np.argmax(mat), mat.shape)
    bi, bj = best
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                               fill=False, edgecolor="black",
                               lw=3.0, zorder=3))
    for i in range(n_rows):
        for j in range(n_cols):
            if (i, j) == (bi, bj):
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold", zorder=4)
            else:
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=8)
    return im, best


def plot_heatmaps(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    layers = data["layers"]
    ways = data["ways"]
    has_gauc = "gauc" in data

    # ═══ 图1: AUC ═══
    fig, ax = plt.subplots(figsize=(8, 6))
    im, best_a = plot_single_heatmap(ax, data["auc"], layers, ways,
                                     "(a) AUC by Layers × Ways", "YlOrRd")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("AUC", fontsize=10)
    cb.outline.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bls_layers_ways_heatmap_auc.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bls_layers_ways_heatmap_auc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] bls_layers_ways_heatmap_auc.pdf")

    # ═══ 图2: GAUC ═══
    if has_gauc:
        fig, ax = plt.subplots(figsize=(8, 6))
        im, best_g = plot_single_heatmap(ax, data["gauc"], layers, ways,
                                         "(b) GAUC by Layers × Ways", "YlGnBu")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("GAUC", fontsize=10)
        cb.outline.set_linewidth(1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bls_layers_ways_heatmap_gauc.pdf"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, "bls_layers_ways_heatmap_gauc.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] bls_layers_ways_heatmap_gauc.pdf")
    else:
        best_g = None

    # ═══ 控制台分析 ═══
    auc = np.array(data["auc"])
    print("\n" + "=" * 60)
    print("层数 × 路数 分析")
    print("=" * 60)
    print(f"  AUC 最佳: 层数={layers[best_a[0]]}, 路数={ways[best_a[1]]}, "
          f"AUC={auc[best_a]:.5f}")
    if has_gauc:
        gauc = np.array(data["gauc"])
        print(f"  GAUC 最佳: 层数={layers[best_g[0]]}, 路数={ways[best_g[1]]}, "
              f"GAUC={gauc[best_g]:.5f}")

    # 行/列均值
    row_mean = auc.mean(axis=1)
    col_mean = auc.mean(axis=0)
    print(f"  最佳层数(按平均): {layers[np.argmax(row_mean)]}")
    print(f"  最佳路数(按平均): {ways[np.argmax(col_mean)]}")

    # 过复杂区检查 (右下角)
    if auc.shape[0] >= 3 and auc.shape[1] >= 3:
        corner = auc[-1, -1]
        print(f"  右下角(最复杂, L={layers[-1]} × {ways[-1]}): {corner:.5f} "
              f"({'低于最佳' if corner < auc.max() else '仍是最佳'})")
        print(f"  与最佳差距: {auc.max() - corner:+.5f}")

    # 边际: 每加一路(固定层数)的平均 ΔAUC
    print("\n  加路边际收益 (每列均值相对 2-way):")
    for j, w in enumerate(ways):
        col = auc[:, j]
        gain = col.mean() - auc[:, 0].mean()
        print(f"    {w:<8}: mean AUC={col.mean():.5f}  Δvs 2-way={gain:+.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True,
                        help="JSON: layers/ways/auc(/gauc)")
    parser.add_argument("--out_dir", type=str, default="./analysis_output")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    assert "layers" in data and "ways" in data and "auc" in data
    assert len(data["auc"]) == len(data["layers"]), "auc 行数必须等于层数"
    assert all(len(r) == len(data["ways"]) for r in data["auc"]), \
        "auc 列数必须等于路数"
    if "gauc" in data:
        assert len(data["gauc"]) == len(data["layers"])
        assert all(len(r) == len(data["ways"]) for r in data["gauc"])
    plot_heatmaps(data, args.out_dir)


if __name__ == "__main__":
    main()
