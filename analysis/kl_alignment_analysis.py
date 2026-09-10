"""
==================================================================
KL 散度多模态对齐深度分析
==================================================================
输入：两个 attn_export .pt 文件（No KL / With KL）
输出：2×4 综合分析图 + 控制台统计汇总

分析维度:
  (a,b) 散点图: ID_attn vs Image_attn（Head绿/Tail红）
  (c)   对角线偏差分布 (σ 收缩 = 更紧对齐)
  (d)   Spearman ρ 分布 (排序一致性)
  (e)   Top-K 重叠率 (K=1,3,5,10,20,30)
  (f)   频次分桶对齐误差 (Cold→Head)
  (g)   注意力熵对比 (防止注意力坍缩)
  (h)   KL Reduction by Frequency Bucket

用法:
python kl_alignment_analysis.py  --attn_no_kl    /root/autodl-fs/ckpt-总/kl/attn_export_only-sata.pt   --attn_with_kl /root/autodl-fs/ckpt-总/kl/attn_export_full.pt  --out_dir /root/autodl-fs/log-总/analysis/output
==================================================================
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ═══════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════

def compute_spearman_per_sample(id_attn, img_attn):
    """每样本 Spearman ρ。跳过常数样本避免 ConstantInputWarning。"""
    B = id_attn.shape[0]
    rhos = np.full(B, np.nan)
    for i in range(B):
        id_v = id_attn[i].numpy()
        img_v = img_attn[i].numpy()
        mask = (id_v > 0) | (img_v > 0)
        if mask.sum() >= 3:
            # 跳过常数样本（标准差为0时 Spearman 无定义）
            if np.std(id_v[mask]) < 1e-10 or np.std(img_v[mask]) < 1e-10:
                continue
            rhos[i], _ = spearmanr(id_v[mask], img_v[mask])
    return rhos

def compute_topk_overlap(id_attn, img_attn, k=10):
    """Image Top-K 中有多少比例也在 ID Top-K 中。"""
    B = id_attn.shape[0]
    overlaps = np.zeros(B)
    actual_k = min(k, id_attn.shape[1])
    for i in range(B):
        id_topk = set(torch.topk(id_attn[i], k=actual_k).indices.numpy())
        img_topk = set(torch.topk(img_attn[i], k=actual_k).indices.numpy())
        overlaps[i] = len(id_topk & img_topk) / actual_k
    return overlaps

def compute_kl_per_sample(id_attn, img_attn, eps=1e-8):
    """KL(img || id) per sample。"""
    B = id_attn.shape[0]
    kls = np.zeros(B)
    for i in range(B):
        p = img_attn[i].numpy()
        q = id_attn[i].numpy()
        mask = p > 0
        kls[i] = np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps)))
    return kls

def compute_entropy(attn_dist):
    """每样本注意力熵。"""
    eps = 1e-8
    return -(attn_dist * torch.log(attn_dist + eps)).sum(dim=1).numpy()

def compute_collapse_ratio(attn_dist, top_k=3, threshold=0.8):
    """注意力坍缩比例。"""
    top_weights = attn_dist.topk(top_k, dim=1).values.sum(dim=1)
    return (top_weights > threshold).float().mean().item()

# ═══════════════════════════════════════════════════════════════
# 频次模拟
# ═══════════════════════════════════════════════════════════════

def simulate_frequencies(item_ids, seed=42):
    """Zipf 分布模拟长尾频次。"""
    np.random.seed(seed)
    unique_ids = np.unique(item_ids.flatten().numpy())
    raw_freqs = np.random.zipf(1.8, len(unique_ids))
    return {int(uid): max(1, int(f)) for uid, f in zip(unique_ids, raw_freqs)}

def get_frequency_bucket(item_ids_flat, freq_map, n_buckets=5):
    """将物品 ID 映射到频次桶。处理频次集中导致的空桶问题。"""
    flat = item_ids_flat.flatten().numpy()
    freqs = np.array([freq_map.get(int(i), 1) for i in flat])
    
    # 检查频次是否有足够的多样性
    unique_freqs = np.unique(freqs[freqs > 0])
    if len(unique_freqs) < n_buckets:
        # 频次太集中，退化为按 ID 均匀分桶
        print(f"  WARNING: Only {len(unique_freqs)} unique frequencies, "
              f"falling back to uniform ID buckets")
        percentiles = np.percentile(flat, np.linspace(0, 100, n_buckets + 1))
        # 去重百分位，确保边界严格递增
        percentiles = np.unique(percentiles)
        buckets = np.clip(np.digitize(flat, percentiles) - 1, 0, n_buckets - 1)
    else:
        percentiles = np.percentile(freqs[freqs > 0], np.linspace(0, 100, n_buckets + 1))
        # 去重百分位，确保边界严格递增
        percentiles = np.unique(percentiles)
        buckets = np.clip(np.digitize(freqs, percentiles) - 1, 0, n_buckets - 1)
    return freqs, buckets

# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════

def _finalize_and_save(fig, out_dir, stem):
    """保存为 PDF 矢量图，并附带 PNG 预览。"""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {pdf_path}")


def plot_full_analysis(data_no_kl, data_with_kl, freq_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    id_no = data_no_kl["id_attn"]
    img_no = data_no_kl["img_attn"]
    id_with = data_with_kl["id_attn"]
    img_with = data_with_kl["img_attn"]

    id_flat_no = id_no.reshape(-1).numpy()
    img_flat_no = img_no.reshape(-1).numpy()
    id_flat_with = id_with.reshape(-1).numpy()
    img_flat_with = img_with.reshape(-1).numpy()

    freqs_no, bucket_no = get_frequency_bucket(data_no_kl["seq_ids"], freq_map)
    freqs_with, bucket_with = get_frequency_bucket(data_with_kl["seq_ids"], freq_map)

    q20 = np.percentile(freqs_no[freqs_no > 0], 20)
    q80 = np.percentile(freqs_no[freqs_no > 0], 80)

    bucket_names = ["Cold", "Tail-2", "Mid", "Head-2", "Head"]
    n_buckets = len(bucket_names)

    # ═══ (a) No KL 散点图 ═══
    fig, ax = plt.subplots(figsize=(6, 6))
    _plot_scatter(ax, id_flat_no, img_flat_no, freqs_no, q20, q80,
                  "(a) No KL: ID vs Image Attention", fixed_lims=[0, 0.08])
    _finalize_and_save(fig, out_dir, "kl_a_no_kl_scatter")

    # ═══ (b) With KL 散点图 ═══
    fig, ax = plt.subplots(figsize=(6, 6))
    _plot_scatter(ax, id_flat_with, img_flat_with, freqs_with, q20, q80,
                  "(b) With KL: ID vs Image Attention")
    _finalize_and_save(fig, out_dir, "kl_b_with_kl_scatter")

    # ═══ (c) 对角线偏差分布 ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    dev_no = id_flat_no - img_flat_no
    dev_with = id_flat_with - img_flat_with
    ax.hist(dev_no, bins=80, alpha=0.5, density=True, color="coral",
            label=f"No KL (σ={np.std(dev_no):.4f})", edgecolor="white")
    ax.hist(dev_with, bins=80, alpha=0.5, density=True, color="steelblue",
            label=f"With KL (σ={np.std(dev_with):.4f})", edgecolor="white")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    # 用 1%~99% 分位限制 x 轴，避免离群点拉宽坐标、主分布被压扁
    dev_all = np.concatenate([dev_no, dev_with])
    x_lo, x_hi = np.percentile(dev_all, [1, 99])
    pad_x = (x_hi - x_lo) * 0.08
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_xlabel("ID_attn − Image_attn"); ax.set_ylabel("Density")
    ax.set_title("(c) Deviation Distribution (narrower = tighter alignment)")
    ax.legend(fontsize=8)
    _finalize_and_save(fig, out_dir, "kl_c_deviation_distribution")

    # ═══ (d) Spearman ρ 分布 ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    rho_no = compute_spearman_per_sample(id_no, img_no)
    rho_with = compute_spearman_per_sample(id_with, img_with)
    valid_no = ~np.isnan(rho_no); valid_with = ~np.isnan(rho_with)
    ax.hist(rho_no[valid_no], bins=40, alpha=0.5, density=True, color="coral",
            label=f"No KL (μ={np.nanmean(rho_no):.3f})", edgecolor="white")
    ax.hist(rho_with[valid_with], bins=40, alpha=0.5, density=True, color="steelblue",
            label=f"With KL (μ={np.nanmean(rho_with):.3f})", edgecolor="white")
    ax.set_xlabel("Spearman ρ per sample"); ax.set_ylabel("Density")
    ax.set_title("(d) Rank Correlation Distribution")
    ax.legend(fontsize=8)
    _finalize_and_save(fig, out_dir, "kl_d_rank_correlation")

    # ═══ (e) Top-K 重叠率 ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = [1, 3, 5, 10, 20, 30]
    overlap_no = [compute_topk_overlap(id_no, img_no, k).mean() for k in ks]
    overlap_with = [compute_topk_overlap(id_with, img_with, k).mean() for k in ks]
    ax.plot(ks, overlap_no, "o-", color="coral", lw=2.5, ms=9, label="No KL")
    ax.plot(ks, overlap_with, "s-", color="steelblue", lw=2.5, ms=9, label="With KL")
    ax.fill_between(ks, overlap_no, overlap_with, alpha=0.12, color="green")
    ax.set_xlabel("K"); ax.set_ylabel("Top-K Overlap Ratio")
    ax.set_title("(e) Top-K Agreement: Do ID & Image agree on top items?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _finalize_and_save(fig, out_dir, "kl_e_topk_agreement")

    # ═══ (f) 频次分桶对齐误差 ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    err_no = np.zeros(n_buckets); err_with = np.zeros(n_buckets)
    for b in range(n_buckets):
        mask_no = bucket_no == b; mask_with = bucket_with == b
        if mask_no.sum() > 0:
            err_no[b] = np.abs(id_flat_no[mask_no] - img_flat_no[mask_no]).mean()
        if mask_with.sum() > 0:
            err_with[b] = np.abs(id_flat_with[mask_with] - img_flat_with[mask_with]).mean()
    x = np.arange(n_buckets); w = 0.35
    ax.bar(x - w/2, err_no, w, color="coral", alpha=0.85, label="No KL", edgecolor="white")
    ax.bar(x + w/2, err_with, w, color="steelblue", alpha=0.85, label="With KL", edgecolor="white")
    for i in range(n_buckets):
        imp = err_no[i] - err_with[i]
        ax.annotate(f"↓{imp:.6f}", (x[i], max(err_no[i], err_with[i]) + 0.0003),
                    ha="center", fontsize=7, fontweight="bold", color="darkgreen")
    ax.set_xlabel("Item Frequency Bucket"); ax.set_ylabel("Mean |ID − Image|")
    ax.set_title("(f) Alignment Error by Frequency Bucket (↓ = improvement)")
    ax.set_xticks(x); ax.set_xticklabels(bucket_names); ax.legend(fontsize=8)
    _finalize_and_save(fig, out_dir, "kl_f_alignment_error")

    # ═══ (g) 注意力熵对比 ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    ent_id_no = compute_entropy(id_no); ent_img_no = compute_entropy(img_no)
    ent_id_with = compute_entropy(id_with); ent_img_with = compute_entropy(img_with)
    positions = [1, 2, 4, 5]
    data_ent = [ent_id_no, ent_img_no, ent_id_with, ent_img_with]
    colors_ent = ["coral", "lightgray", "steelblue", "lightgray"]
    bp = ax.boxplot(data_ent, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors_ent):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(["ID\n(No KL)", "Image\n(No KL)", "ID\n(With KL)", "Image\n(With KL)"], fontsize=8)
    ax.set_ylabel("Attention Entropy")
    ax.set_title("(g) Entropy: KL prevents attention collapse")
    _finalize_and_save(fig, out_dir, "kl_g_attention_entropy")

    # ═══ 控制台汇总 ═══
    print("\n" + "=" * 60)
    print("KL Alignment Analysis — Summary")
    print("=" * 60)
    print(f"  Spearman ρ (No KL):     μ={np.nanmean(rho_no):.4f}  σ={np.nanstd(rho_no):.4f}")
    print(f"  Spearman ρ (With KL):   μ={np.nanmean(rho_with):.4f}  σ={np.nanstd(rho_with):.4f}")
    print(f"  Top-5 Overlap (No KL):   {overlap_no[2]:.4f}")
    print(f"  Top-5 Overlap (With KL): {overlap_with[2]:.4f}")
    print(f"  Alignment Error by Bucket:")
    for b, name in enumerate(bucket_names):
        print(f"    {name:<10}: No KL={err_no[b]:.6f}  With KL={err_with[b]:.6f}  "
              f"Δ={err_no[b]-err_with[b]:.6f}")


def _plot_scatter(ax, id_flat, img_flat, freqs_flat, q20, q80, title, fixed_lims=None):
    """带频次颜色的散点图。
    默认按数据云动态缩放；若传入 fixed_lims=[lo, hi] 则使用固定轴范围。"""
    n_sample = min(15000, len(id_flat))
    idx = np.random.choice(len(id_flat), n_sample, replace=False)
    xs, ys, fs = img_flat[idx], id_flat[idx], freqs_flat[idx]
    for cat, c, s, lbl in [
        (fs <= q20, "#e74c3c", 3, "Tail"),
        ((fs > q20) & (fs < q80), "#bdc3c7", 1, "Mid"),
        (fs >= q80, "#2ecc71", 3, "Head"),
    ]:
        if cat.sum():
            ax.scatter(xs[cat], ys[cat], c=c, s=s, alpha=0.35, label=lbl, edgecolors="none")
    if fixed_lims is not None:
        # ── 固定轴范围 ──
        lo, hi = fixed_lims[0], fixed_lims[1]
    else:
        # ── 轴范围：按数据云动态缩放，去掉从 0 起步导致的空白角落 ──
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        pad = max(x_max - x_min, y_max - y_min) * 0.1 + 1e-9
        lo = min(x_min, y_min) - pad
        hi = max(x_max, y_max) + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Image Attention"); ax.set_ylabel("ID Attention")
    ax.set_title(title); ax.legend(fontsize=7, markerscale=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════
# 方案二: KL 收益分解
# ═══════════════════════════════════════════════════════════════

def decompose_kl_benefit(id_no_kl, id_with_kl, img_attn):
    """
    方案二: KL 收益分解。
    对每个序列位置计算 Δattn = |id_with_kl - id_no_kl|,
    然后计算 Spearman ρ(Δattn, img_attn) 看 KL 是否把注意力推向多模态位置。
    返回 (rho_per_sample, delta_magnitude_per_sample)
    """
    B = id_no_kl.shape[0]
    rhos = np.full(B, np.nan)
    delta_mags = np.zeros(B)

    for i in range(B):
        delta = np.abs(id_with_kl[i].numpy() - id_no_kl[i].numpy())
        mm = img_attn[i].numpy()
        mask = delta > 1e-8
        delta_mags[i] = delta.mean()
        if mask.sum() >= 3:
            # 跳过常数样本
            if np.std(delta[mask]) < 1e-10 or np.std(mm[mask]) < 1e-10:
                continue
            rhos[i], _ = spearmanr(delta[mask], mm[mask])

    return rhos, delta_mags


# ═══════════════════════════════════════════════════════════════
# 方案四: 训练线性 probe 测试注意力特征的预测质量
# ═══════════════════════════════════════════════════════════════

def probe_attention_quality(id_attn, labels, train_ratio=0.7):
    """
    方案四: 训练线性 probe 测试注意力特征的预测质量。
    只用 ID 注意力分布作为输入，预测 label。
    AUC 越高 → 注意力中包含的预测信息越多。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    n = id_attn.shape[0]
    split = int(n * train_ratio)
    idx = np.random.permutation(n)

    X = id_attn[idx].numpy()  # [N, seq_len]
    y = labels[idx, 1].numpy()  # positive class probability

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    clf = LogisticRegression(max_iter=1000, C=1.0).fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)

    return auc


# ═══════════════════════════════════════════════════════════════
# 方案四: 训练线性 probe 测试注意力特征的预测质量
# ═══════════════════════════════════════════════════════════════

def plot_deeper_analysis(data_no_kl, data_with_kl, freq_map, out_dir):
    """方案二的深层分析图（单独 PDF 输出）。"""
    os.makedirs(out_dir, exist_ok=True)

    id_no = data_no_kl["id_attn"]
    img_no = data_no_kl["img_attn"]
    id_with = data_with_kl["id_attn"]

    rho_kl, _ = decompose_kl_benefit(id_no, id_with, img_no)
    valid_rho = rho_kl[~np.isnan(rho_kl)]

    # ═══ (a) KL 收益分解分布 ═══
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid_rho, bins=40, color="steelblue", alpha=0.8, edgecolor="white", density=True)
    ax.axvline(np.mean(valid_rho), color="red", ls="--", lw=2,
               label=f"mean={np.mean(valid_rho):.3f}")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Spearman ρ(Δattn, Image_sim) per sample")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) KL Benefit Decomposition\nρ>0 ratio: {(valid_rho>0).mean():.1%}")
    ax.legend(fontsize=9)
    _finalize_and_save(fig, out_dir, "kl_d1_benefit_decomposition")

    print(f"\n{'='*60}")
    print("KL Deeper Analysis — Summary")
    print(f"{'='*60}")
    print(f"  KL Benefit ρ (mean):  {np.mean(valid_rho):.4f}")
    print(f"  KL Benefit ρ>0:       {(valid_rho>0).mean():.1%}")


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="KL alignment deep analysis")
    parser.add_argument("--attn_no_kl", type=str, required=True)
    parser.add_argument("--attn_with_kl", type=str, required=True)
    parser.add_argument("--freq_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./analysis_output")
    args = parser.parse_args()

    data_no_kl = torch.load(args.attn_no_kl, map_location="cpu", weights_only=False)
    data_with_kl = torch.load(args.attn_with_kl, map_location="cpu", weights_only=False)
    print(f"Loaded No-KL:  id_attn {data_no_kl['id_attn'].shape}")
    print(f"Loaded With-KL: id_attn {data_with_kl['id_attn'].shape}")

    if args.freq_file and os.path.exists(args.freq_file):
        freq_arr = np.load(args.freq_file)
        if freq_arr.ndim == 1:
            freq_map = {i: int(f) for i, f in enumerate(freq_arr)}
        else:
            freq_map = {int(r[0]): int(r[1]) for r in freq_arr}
        print(f"Loaded frequency map: {len(freq_map)} items")
    else:
        print("No frequency file — using Zipf simulation")
        freq_map = simulate_frequencies(data_no_kl["seq_ids"])

    plot_full_analysis(data_no_kl, data_with_kl, freq_map, args.out_dir)
    try:
        plot_deeper_analysis(data_no_kl, data_with_kl, freq_map, args.out_dir)
    except ImportError as e:
        print(f"\n[Skipping deeper analysis] sklearn not available: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
