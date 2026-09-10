"""
==================================================================
双轴折线图: 粒度 vs AUC / GAUC
==================================================================
图 1: 单粒度直接对比 (6 个粒度 eps=0.1~0.6 各单独训练)
      左轴 AUC, 右轴 GAUC, 共 x 轴 = eps
图 2: 多分辨率累积对比 (multi_configs 6 组配置)
      左轴 AUC, 右轴 GAUC, 共 x 轴 = 分辨率数量

用法:
    1. 把下面 DATA 区里的示例数值替换成你的真实 AUC / GAUC
    2. 运行:
    python plot_perf_dual_axis.py  --out_dir /root/autodl-fs/log-总/analysis/output
输出 (默认 PDF 矢量图, 可用 --format 切换):
    analysis_output/perf_single_granularity.pdf   (图 1)
    analysis_output/perf_multi_resolution.pdf     (图 2)
==================================================================
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================
#  DATA 区 —— 请替换为真实数据 (当前为示例数值, 仅用于预览效果)
# ==================================================================

# ── 图 1: 单粒度, 6 个粒度各单独训练 ────────────────────────────
SINGLE_EPS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SINGLE_AUC  = [0.644368, 0.643345, 0.643908, 0.643474, 0.640588, 0.640865]  # TODO: 替换
SINGLE_GAUC = [0.614767, 0.613718, 0.613961, 0.613558, 0.609009, 0.609621]  # TODO: 替换

# ── 图 2: 多分辨率累积, 6 组配置 (与 multi_configs 对应) ────────
# n_res  = 分辨率数量;  auc / gauc 为对应配置的指标
MULTI_CONFIGS = [
    {"name": "abl_res_1", "eps_list": [0.1],                      "n_res": 1, "auc": 0.644368, "gauc": 0.614767},  # TODO: 替换
    {"name": "abl_res_2", "eps_list": [0.1, 0.2],                 "n_res": 2, "auc": 0.644588, "gauc": 0.614908},  # TODO: 替换
    {"name": "abl_res_3", "eps_list": [0.1, 0.2, 0.3],            "n_res": 3, "auc": 0.644313, "gauc": 0.614682},  # TODO: 替换
    {"name": "abl_res_4", "eps_list": [0.1, 0.2, 0.3, 0.4],       "n_res": 4, "auc": 0.644435, "gauc": 0.614726},  # TODO: 替换
    {"name": "abl_res_5", "eps_list": [0.1, 0.2, 0.3, 0.4, 0.6],  "n_res": 5, "auc": 0.644364, "gauc": 0.615171},  # TODO: 替换
    {"name": "abl_res_6", "eps_list": [0.1, 0.2, 0.3, 0.5, 0.6],  "n_res": 6, "auc": 0.644752, "gauc": 0.615066},  # TODO: 替换
    {"name": "abl_res_7", "eps_list": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "n_res": 7, "auc": 0.644515, "gauc": 0.614916},  # TODO: 替换
]

# ==================================================================
#  绘图
# ==================================================================
def _fmt_eps(eps_list):
    """把 eps 列表格式化为紧凑字符串，如 [0.1,0.2,0.3] -> '0.1 0.2 0.3'。"""
    return " ".join(f"{e:.1f}" for e in eps_list)


def plot_dual_axis(x, y_left, y_right, x_label, title, save_path,
                   left_label="AUC", right_label="GAUC",
                   x_ticks=None, x_tick_labels=None, left_color="#1f77b4",
                   right_color="#d62728", left_marker="o", right_marker="s",
                   x_tick_fontsize=10, x_tick_rotation=0, figsize=(9, 6),
                   show_xlabel=True, show_best_annot=True):
    """
    通用双轴折线图: 左轴 y_left, 右轴 y_right, 共享 x。
    标注左/右各自的最优值。
    show_xlabel: 是否显示 x 轴标签文字
    show_best_annot: 是否显示 "Best ..." 最优值标注
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax_left = plt.subplots(figsize=figsize)

    # ── 左轴: AUC ──
    line_left, = ax_left.plot(
        x, y_left, color=left_color, marker=left_marker, markersize=7,
        linewidth=2.2, label=left_label,
    )
    if show_xlabel:
        ax_left.set_xlabel(x_label, fontsize=12)
    ax_left.set_ylabel(left_label, color=left_color, fontsize=12)
    ax_left.tick_params(axis="y", labelcolor=left_color)
    ax_left.grid(True, alpha=0.3)

    # 左轴最优值标注
    if show_best_annot:
        i_best_l = int(np.argmax(y_left))
        ax_left.annotate(
            f"Best {left_label}: {y_left[i_best_l]:.6f}",
            (x[i_best_l], y_left[i_best_l]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, color=left_color,
            arrowprops=dict(arrowstyle="->", color=left_color, lw=1.0),
        )

    # ── 右轴: GAUC ──
    ax_right = ax_left.twinx()
    line_right, = ax_right.plot(
        x, y_right, color=right_color, marker=right_marker, markersize=7,
        linewidth=2.2, linestyle="--", label=right_label,
    )
    ax_right.set_ylabel(right_label, color=right_color, fontsize=12)
    ax_right.tick_params(axis="y", labelcolor=right_color)

    # 右轴最优值标注
    if show_best_annot:
        i_best_r = int(np.argmax(y_right))
        ax_right.annotate(
            f"Best {right_label}: {y_right[i_best_r]:.6f}",
            (x[i_best_r], y_right[i_best_r]),
            xytext=(0, -18), textcoords="offset points",
            ha="center", fontsize=9, color=right_color,
            arrowprops=dict(arrowstyle="->", color=right_color, lw=1.0),
        )

    # x 轴刻度
    if x_ticks is not None:
        ax_left.set_xticks(x_ticks)
        if x_tick_labels is not None:
            ax_left.set_xticklabels(x_tick_labels, fontsize=x_tick_fontsize)
            # 旋转刻度标签避免长标签重叠
            plt.setp(ax_left.get_xticklabels(), rotation=x_tick_rotation,
                     ha="right" if x_tick_rotation else "center")
    # 避免多行刻度被截断
    ax_left.tick_params(axis="x", pad=5)

    # 合并图例
    lines = [line_left, line_right]
    ax_left.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=10)

    ax_left.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Figure saved] {save_path}")
    plt.close()


def plot_single_granularity(out_dir, fmt="pdf"):
    """图 1: 单粒度 6 个 eps 的 AUC / GAUC 双轴折线图。
    每个 eps 刻度下标注对应的分桶数 n_bins = 2*int(1/eps)+2。"""
    save_path = os.path.join(out_dir, f"perf_single_granularity.{fmt}")

    # 每个 eps 对应的桶数 (公式: 2*int(1/eps)+2)
    bucket_counts = [2 * int(1.0 / e) + 2 for e in SINGLE_EPS]

    plot_dual_axis(
        x=SINGLE_EPS,
        y_left=SINGLE_AUC,
        y_right=SINGLE_GAUC,
        x_label="Granularity $\\epsilon$ (finer → coarser)",
        title="(a) Single Granularity: AUC & GAUC vs $\\epsilon$",
        save_path=save_path,
        x_ticks=SINGLE_EPS,
        x_tick_labels=[f"{e:.1f}\n({n} bins)" for e, n in zip(SINGLE_EPS, bucket_counts)],
        show_xlabel=True,      # 不显示底部 x 轴文字
        show_best_annot=False,  # 不显示 Best 最优值标注
    )

    # 打印单粒度数值表 (含桶数)
    print("\n── 单粒度 数值表 ──")
    print(f"{'eps':>5} {'bins':>6} {'AUC':>10} {'GAUC':>10}")
    for e, n, a, g in zip(SINGLE_EPS, bucket_counts, SINGLE_AUC, SINGLE_GAUC):
        print(f"{e:>5.2f} {n:>6} {a:>10.6f} {g:>10.6f}")


def plot_multi_resolution(out_dir, fmt="pdf"):
    """图 2: 多分辨率累积 6 组配置的 AUC / GAUC 双轴折线图。
    x 轴刻度两行: 第一行分辨率数量, 第二行对应的 eps 粒度组合。"""
    save_path = os.path.join(out_dir, f"perf_multi_resolution.{fmt}")

    n_res = [c["n_res"] for c in MULTI_CONFIGS]
    auc = [c["auc"] for c in MULTI_CONFIGS]
    gauc = [c["gauc"] for c in MULTI_CONFIGS]
    names = [c["name"] for c in MULTI_CONFIGS]

    # x 轴刻度: 只显示 ε 粒度组合 (与图 1 风格一致, 更清爽)
    x_tick_labels = [
        f"\u03b5={_fmt_eps(c['eps_list'])}" for c in MULTI_CONFIGS
    ]

    plot_dual_axis(
        x=n_res,
        y_left=auc,
        y_right=gauc,
        x_label="Number of Resolutions (cumulative, with $\\epsilon$ combination)",
        title="(b) Multi-Resolution: AUC & GAUC vs #Resolutions",
        save_path=save_path,
        x_ticks=n_res,
        x_tick_labels=x_tick_labels,
        x_tick_fontsize=10,     # 与图 1 相同字号, 更清晰
        x_tick_rotation=30,     # 倾斜 30°, 避免长标签重叠
        figsize=(12, 6),        # 加宽容纳长标签
        show_xlabel=False,      # 不显示底部 x 轴文字
        show_best_annot=False,  # 不显示 Best 最优值标注
    )

    # 打印多分辨率数值表 (含 eps 组合)
    print("\n── 多分辨率累积 数值表 ──")
    print(f"{'Config':<12} {'eps_list':>38} {'n_res':>6} {'AUC':>10} {'GAUC':>10}")
    for c in MULTI_CONFIGS:
        eps_str = str([f"{e:.1f}" for e in c["eps_list"]])
        print(f"{c['name']:<12} {eps_str:>38} {c['n_res']:>6} {c['auc']:>10.6f} {c['gauc']:>10.6f}")


def main():
    parser = argparse.ArgumentParser(description="Dual-axis AUC/GAUC plots")
    parser.add_argument("--out_dir", type=str, default="./analysis_output",
                        help="Output directory for plots")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"],
                        help="Output figure format (default: pdf vector)")
    args = parser.parse_args()

    print("=" * 60)
    print("AUC / GAUC Dual-Axis Analysis")
    print("=" * 60)

    plot_single_granularity(args.out_dir, args.format)
    plot_multi_resolution(args.out_dir, args.format)

    print("\nDone.")


if __name__ == "__main__":
    main()
