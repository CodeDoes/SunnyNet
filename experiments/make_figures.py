"""生成架构示意图与实验图表 (matplotlib, 无外部绘图依赖).

产出 docs/figures/:
  architecture.png   状态传递与块内数据流示意
  speed_compare.png  基线对照: 每 token 耗时 / 吞吐 / 峰值内存
  loss_curves.png    训练曲线 (train/val)
  ablation.png       消融结果
  seq_scaling.png    前向耗时 vs 序列长度 (检验 O(N))
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
RES = ROOT / "experiments" / "results"

_CJK_OK = False

INK = "#1b1f24"
MUTED = "#6b7280"
ACCENT = "#2f6fb5"
WARM = "#c2410c"
LINE = "#d1d5db"


def _style():
    """配置字体与样式. 中文字体需实际渲染验证, 不能只靠名称匹配."""
    global _CJK_OK
    from matplotlib import font_manager
    prefer = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Noto Sans CJK SC",
              "Source Han Sans SC", "DengXian", "SimSun", "STSong"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in prefer if n in have), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        _CJK_OK = _probe_glyph(chosen)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#eceff3",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })


def _probe_glyph(font_name: str) -> bool:
    """渲染一个汉字并检查是否画出真实字形 (未命中会得到空白/豆腐块)."""
    try:
        fig = plt.figure(figsize=(1, 1))
        fig.text(0.1, 0.3, "状态", fontname=font_name, fontsize=24)
        fig.canvas.draw()
        buf = __import__("numpy").asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return bool((buf < 200).sum() > 60)
    except Exception:
        return False


def _has_cjk() -> bool:
    return _CJK_OK


def L(zh: str, en: str) -> str:
    return zh if _CJK_OK else en


def box(ax, x, y, w, h, text, fc="#f8fafc", ec=LINE, fs=9, weight="normal",
        tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, p1, p2, color=ACCENT, style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 shrinkA=1.5, shrinkB=1.5))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ax.text(5.1, 6.32, L("SunnyNet 状态传递与块内数据流", "SunnyNet: state pass & block dataflow"),
            ha="center", fontsize=12.5, weight="bold", color=INK)
    ax.text(5.1, 6.02, L("O(N) 线性复杂度 · 滚动状态接力 · 无残差 · 状态反馈后置",
                         "O(N) complexity · rolling state relay · no residual · post-scan feedback"),
            ha="center", fontsize=9, color=MUTED)

    box(ax, 0.35, 5.32, 1.5, 0.46, L("输入 token 流", "input tokens"), fc="#eef2f7", fs=9)
    arrow(ax, (1.85, 5.55), (2.4, 5.55))

    # Block 1
    box(ax, 2.4, 2.15, 4.5, 3.62, "", fc="#fbfcfd", ec="#b9c3cd", fs=9)
    ax.text(4.65, 5.55, L("Block 1", "Block 1"), ha="center", fontsize=10.5,
            weight="bold", color=INK)

    box(ax, 2.62, 4.86, 1.42, 0.46, L("CSN 归一化", "CSN norm"), fs=8.5)
    box(ax, 4.2, 4.86, 1.42, 0.46, L("深度可分因果卷积", "depthwise causal conv"), fs=8)
    arrow(ax, (4.04, 5.09), (4.2, 5.09))
    box(ax, 5.78, 4.86, 0.94, 0.46, L("低秩投影\nU(V(·))", "low-rank\nU(V(·))"), fs=8)
    arrow(ax, (5.62, 5.09), (5.78, 5.09))

    box(ax, 2.62, 4.06, 1.42, 0.5, L("候选状态 u\n(u 与状态无关)", "candidate u\n(independent of s)"), fs=8)
    arrow(ax, (6.25, 4.86), (3.33, 4.56), ls="--")
    box(ax, 4.2, 4.06, 1.42, 0.5, L("组内竞争门控\nw∈(0,1)", "grouped gate\nw∈(0,1)"), fs=8)
    arrow(ax, (4.04, 4.31), (4.2, 4.31))
    box(ax, 5.78, 4.06, 0.94, 0.5, L("保护门 p\n∈[1-δ,1]", "keep gate p\n∈[1-δ,1]"), fs=8)
    arrow(ax, (5.62, 4.31), (5.78, 4.31))

    box(ax, 2.62, 3.2, 4.1, 0.55,
        L("对数域并行扫描\nS = Scan(u, p, s_in)   块内两次 cumsum",
          "log-space parallel scan\nS = Scan(u, p, s_in)   two cumsums / chunk"),
        fc="#eef6ff", ec=ACCENT, fs=8.2, weight="bold")
    arrow(ax, (4.91, 4.06), (4.91, 3.75))

    box(ax, 2.62, 2.36, 4.1, 0.62,
        L("输出  o = W_c·y + β·S + W_s·circ(S_prev)\n状态反馈在扫描之后 → 因果 + 分段严格等价",
          "out  o = W_c·y + β·S + W_s·circ(S_prev)\nfeedback after scan → causal & segment-exact"),
        fc="#fff7ed", ec=WARM, fs=8)
    arrow(ax, (4.91, 3.2), (4.91, 2.98))

    # 接力
    arrow(ax, (6.9, 4.0), (7.72, 4.0), color=ACCENT, lw=1.6)
    ax.text(7.31, 3.95, L("块末状态 S_T → 下一块 s_0", "S_T → next s_0"),
            ha="center", va="top", rotation=90, fontsize=7.8, color=ACCENT)
    box(ax, 7.72, 2.15, 2.13, 3.62, "", fc="#fbfcfd", ec="#b9c3cd")
    ax.text(8.78, 5.55, L("Block 2 … L", "Block 2 … L"), ha="center",
            fontsize=10.5, weight="bold", color=INK)
    ax.text(8.78, 3.9, L("同构重复\n\n状态接力\n\n无加法残差\n\n特征流并行传递",
                        "same structure\n\nstate relay\n\nno additive residual\n\nfeature stream in parallel"),
            ha="center", va="center", fontsize=8.4, color=MUTED, linespacing=1.6)

    arrow(ax, (9.85, 3.96), (10.12, 3.96))
    box(ax, 7.72, 1.5, 2.13, 0.46, L("终态归一化 + LM Head", "final norm + LM Head"), fc="#eef2f7", fs=8.5)
    arrow(ax, (8.78, 2.15), (8.78, 1.96))

    ax.text(5.1, 1.12, L("状态更新（凸组合，天然有界）：", "state update (convex combination, bounded):"),
            ha="center", fontsize=9, color=INK)
    ax.text(5.1, 0.72, r"$s_t = p_t \odot s_{t-1} + (1-p_t)\odot u_t,\quad p_t\in[1-\delta_i,\,1]$",
            ha="center", fontsize=11, color=ACCENT)
    ax.text(5.1, 0.3, L(r"记忆跨度 $H_i=1/\delta_i$ 在对数刻度遍历 [2, 8192] → 单层内多时间尺度",
                        r"memory span $H_i=1/\delta_i$ log-uniform over [2, 8192] → multi-scale in one layer"),
            ha="center", fontsize=8.4, color=MUTED)

    fig.savefig(FIG / "architecture.png")
    plt.close(fig)


def fig_speed_compare():
    p = RES / "report_baseline.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text(encoding="utf-8"))
    runs = [x for x in r["runs"] if "error" not in x]
    if len(runs) < 2:
        return False
    labels = [x["tag"] for x in runs]
    ms = [x["summary"]["ms_per_token"] for x in runs]
    tps = [x["summary"]["tokens_per_sec"] for x in runs]
    rss = [x["summary"]["peak_rss_mb"] for x in runs]
    ppl = [x["summary"]["best_ppl"] for x in runs]
    params = [x["summary"]["params_non_embed"] for x in runs]
    colors = [ACCENT, WARM][:len(runs)]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    panels = [
        (ms, L("每 token 训练耗时 (ms)", "ms / token (train)"), True),
        (tps, L("吞吐 (token/s)", "throughput (tok/s)"), False),
        (rss, L("峰值内存 (MB)", "peak RSS (MB)"), True),
        (ppl, L("验证困惑度 (越低越好)", "val perplexity (lower better)"), True),
    ]
    for ax, (vals, title, lower) in zip(axes, panels):
        b = ax.bar(labels, vals, color=colors, width=0.55)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        best = min(vals) if lower else max(vals)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,.4g}",
                    ha="center", va="bottom", fontsize=8.6, color=INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.text(0.5, 0.94, L("← 越低越好", "lower is better") if lower else L("← 越高越好", "higher is better"),
                transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    sub = " | ".join(f"{l}: {q/1e6:.3f}M {L('计算栈参数','stack params')}"
                     for l, q in zip(labels, params))
    fig.suptitle(L("SunnyNet vs Transformer 基线对照（同 token 预算 / 同线程数 / 参数量对齐）",
                   "SunnyNet vs Transformer baseline (same tokens / threads / matched params)"),
                 fontsize=11.5, weight="bold", y=1.06)
    fig.text(0.5, -0.04, sub, ha="center", fontsize=8.6, color=MUTED)
    fig.savefig(FIG / "speed_compare.png")
    plt.close(fig)
    return True


def fig_loss_curves():
    p = RES / "report_baseline.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    for x in r["runs"]:
        if "error" in x or not x.get("history"):
            continue
        h = x["history"]
        steps = [e["step"] for e in h]
        axes[0].plot(steps, [e["train_loss"] for e in h], marker="o", ms=3.2,
                     lw=1.6, label=x["tag"])
        axes[1].plot(steps, [e["val_loss"] for e in h], marker="s", ms=3.2,
                     lw=1.6, label=x["tag"])
    for ax, t in zip(axes, [L("训练损失", "train loss"), L("验证损失", "val loss")]):
        ax.set_xlabel(L("步数", "step"))
        ax.set_ylabel(t)
        ax.set_title(t)
        ax.legend(frameon=False)
    fig.suptitle(L("训练曲线", "Training curves"), fontsize=11.5, weight="bold", y=1.02)
    fig.savefig(FIG / "loss_curves.png")
    plt.close(fig)
    return True


def fig_ablation():
    p = RES / "report_ablation.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text(encoding="utf-8"))
    runs = [x for x in r["runs"] if "error" not in x]
    if not runs:
        return False
    names = [x["tag"] for x in runs]
    ms = [x["summary"]["ms_per_token"] for x in runs]
    ppl = [x["summary"]["best_ppl"] for x in runs]
    steps = runs[0]["summary"].get("steps_done")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0))
    for ax, vals, title in (
            (axes[0], ms, L("每 token 耗时 (ms)", "ms/token")),
            (axes[1], ppl, L("验证困惑度", "val perplexity"))):
        b = ax.bar(names, vals, color=[ACCENT] + [MUTED] * (len(names) - 1), width=0.55)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.45, hi * 0.02)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + pad * 0.12, f"{v:,.4g}",
                    ha="center", va="bottom", fontsize=8.6)
        ax.tick_params(axis="x", rotation=22)
    axes[1].text(0.5, -0.34,
                 L(f"注意: 仅 {steps} 步, 全部变体均未收敛, 差异在噪声量级, 不作为组件有效性判据",
                   f"Note: only {steps} steps, none converged; differences within noise, not evidence of component value"),
                 transform=axes[1].transAxes, ha="center", fontsize=8.2, color=WARM)
    fig.suptitle(L("消融实验（full 为完整模型；纵轴已放大以显示差异）",
                   "Ablation (full = complete; y-axis zoomed to show differences)"),
                 fontsize=11.5, weight="bold", y=1.02)
    fig.savefig(FIG / "ablation.png")
    plt.close(fig)
    return True


def fig_longseq_train():
    """长序列训练对照 (T=1024): 速度 / 质量 / 内存三panel."""
    a = RES / "longseq_sunnynet_T1024.json"
    b = RES / "longseq_transformer_T1024.json"
    if not (a.exists() and b.exists()):
        return False
    sa = json.loads(a.read_text(encoding="utf-8"))["summary"]
    sb = json.loads(b.read_text(encoding="utf-8"))["summary"]
    labels = ["sunnynet", "transformer"]
    colors = [ACCENT, WARM]
    panels = [
        ([sa["ms_per_token_mean"], sb["ms_per_token_mean"]],
         L("每 token 耗时 (ms)", "ms/token"), True, "{:.4f}"),
        ([sa["best_ppl"], sb["best_ppl"]],
         L("验证困惑度", "val perplexity"), True, "{:.1f}"),
        ([sa["peak_rss_mb"], sb["peak_rss_mb"]],
         L("峰值内存 (MB)", "peak RSS (MB)"), True, "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for ax, (vals, title, lower, fmt) in zip(axes, panels):
        b_ = ax.bar(labels, vals, color=colors, width=0.55)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        for rect, v in zip(b_, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_ylim(0, max(vals) * 1.2)
    ratio_ms = sb["ms_per_token_mean"] / sa["ms_per_token_mean"]
    ratio_ppl = sb["best_ppl"] / sa["best_ppl"]
    fig.suptitle(L(f"长序列训练对照 T=1024（均优于 Transformer：快 {ratio_ms:.2f}×, ppl 比为 {ratio_ppl:.2f}）",
                   f"Long-sequence train @ T=1024 (SunnyNet wins {ratio_ms:.2f}x speed, ppl ratio {ratio_ppl:.2f})"),
                 fontsize=11.5, weight="bold", y=1.03)
    fig.savefig(FIG / "longseq_train.png")
    plt.close(fig)
    return True


def fig_on_compare():
    """七架构同协议对比: 质量 / 速度 双 panel (按 ppl 排序)."""
    files = {}
    for p in sorted(RES.glob("oncompare_*.json")):
        name = p.stem.replace("oncompare_", "")
        files[name] = json.loads(p.read_text(encoding="utf-8"))
    if len(files) < 2:
        return False
    order = ["sunnynet", "transformer", "linear_attn", "rwkv", "mamba", "hyena", "h3"]
    names = [n for n in order if n in files]
    ppl = [files[n]["summary"]["best_ppl"] for n in names]
    mst = [files[n]["summary"]["ms_per_token_mean"] for n in names]
    colors = [ACCENT if n == "sunnynet" else (WARM if n == "transformer" else MUTED)
              for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2))
    for ax, vals, title, fmt in (
            (axes[0], ppl, L("验证困惑度 (越低越好)", "val perplexity (lower better)"), "{:.0f}"),
            (axes[1], mst, L("每 token 耗时 ms (越低越好)", "ms/token (lower better)"), "{:.3f}")):
        b = ax.bar(names, vals, color=colors, width=0.6)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8.6, color=INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.tick_params(axis="x", rotation=20)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT),
               plt.Rectangle((0, 0), 1, 1, color=WARM),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    fig.legend(handles, [L("SunnyNet", "SunnyNet"), "Transformer",
                         L("同复杂度 O(N) 架构", "O(N) peers")],
               loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(L("同计算栈参数量下的七架构对比（T=256, 600 步）",
                   "Seven architectures at matched stack params (T=256, 600 steps)"),
                 fontsize=11.5, weight="bold", y=1.06)
    fig.savefig(FIG / "on_compare.png")
    plt.close(fig)
    return True


def fig_on_scaling():
    """七架构 ms/token vs 序列长度 (log-log), 检验 O(N) 缩放是否同益同行."""
    p = RES / "on_scaling_t8.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text(encoding="utf-8"))
    order = ["sunnynet", "transformer", "linear_attn", "rwkv", "mamba", "hyena", "h3"]
    names = [n for n in order if n in d]
    lens = [r["seq_len"] for r in d["sunnynet"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for name in names:
        ys = [r["ms_per_token"] for r in d[name]]
        if name == "sunnynet":
            style = dict(color=ACCENT, lw=2.6, ms=6, zorder=5)
        elif name == "transformer":
            style = dict(color=WARM, lw=2.0, ms=5, zorder=4)
        else:
            style = dict(color=MUTED, lw=1.2, ms=3.6, alpha=0.85)
        ax.plot(lens, ys, marker="o", label=name, **style)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(lens)
    ax.set_xticklabels([str(t) for t in lens])
    ax.set_xlabel(L("序列长度 T", "sequence length T"))
    ax.set_ylabel(L("每 token 前向耗时 ms (对数轴)", "ms per token, forward (log)"))
    ax.set_title(L("七架构 ms/token 随序列长度变化（batch=1, 8 线程, 进程隔离）",
                   "ms/token vs T, seven architectures (batch=1, 8 threads, isolated)"),
                 fontsize=10.6, weight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=8.6)
    fig.savefig(FIG / "on_scaling.png")
    plt.close(fig)
    return True


def fig_seq_scaling():
    p = RES / "seq_scaling.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text(encoding="utf-8"))
    names = list(d)
    lens = [r["seq_len"] for r in d[names[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

    ax = axes[0]
    colors = {"sunnynet": ACCENT, "transformer": WARM}
    for name in names:
        ys = [r["ms"] for r in d[name]]
        ax.plot(lens, ys, marker="o", ms=4.5, lw=1.8,
                color=colors.get(name), label=name)
    base = d[names[0]][0]["ms"] / lens[0]
    ax.plot(lens, [base * t for t in lens], ls="--", lw=1.2, color=MUTED,
            label=L("线性参照 O(N)", "linear ref O(N)"))
    ax.set_yscale("log")
    ax.set_xlabel(L("序列长度", "sequence length"))
    ax.set_ylabel(L("单次前向耗时 (ms, 对数轴)", "forward time (ms, log)"))
    ax.set_title(L("前向耗时 vs 序列长度", "Forward time vs sequence length"))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ratio = [d["transformer"][i]["ms"] / d["sunnynet"][i]["ms"] for i in range(len(lens))]
    ax.axhline(1.0, ls="--", lw=1.2, color=MUTED)
    ax.plot(lens, ratio, marker="s", ms=5, lw=1.9, color=ACCENT)
    for x, r in zip(lens, ratio):
        ax.annotate(f"{r:.2f}x", (x, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.4, color=INK)
    cross = None
    for i in range(1, len(lens)):
        if (ratio[i - 1] - 1.0) * (ratio[i] - 1.0) <= 0:
            x0, x1 = lens[i - 1], lens[i]
            y0, y1 = ratio[i - 1], ratio[i]
            cross = x0 + (x1 - x0) * (1.0 - y0) / (y1 - y0)
            break
    if cross:
        ax.axvline(cross, ls=":", lw=1.3, color=WARM)
        ax.text(cross, max(ratio) * 0.55, L(f"交叉点 T≈{cross:.0f}", f"crossover T≈{cross:.0f}"),
                rotation=90, va="center", ha="right", fontsize=8.6, color=WARM)
    ax.set_xlabel(L("序列长度", "sequence length"))
    ax.set_ylabel(L("耗时比 (Transformer / SunnyNet)", "time ratio (TF / Sunny)"))
    ax.set_title(L("相对速度: >1 表示 SunnyNet 更快", "relative speed: >1 means SunnyNet faster"))
    fig.suptitle(L("O(N) 线性复杂度的实测验证", "Measured verification of O(N) linear complexity"),
                 fontsize=11.5, weight="bold", y=1.02)
    fig.savefig(FIG / "seq_scaling.png")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    _style()
    print(f"  中文字形可用: {_CJK_OK}")
    todo = {
        "architecture": fig_architecture,
        "speed": fig_speed_compare,
        "loss": fig_loss_curves,
        "ablation": fig_ablation,
        "scaling": fig_seq_scaling,
        "longseq": fig_longseq_train,
        "oncompare": fig_on_compare,
        "onscaling": fig_on_scaling,
    }
    for k, fn in todo.items():
        if args.only and args.only != k:
            continue
        try:
            ok = fn() if k != "architecture" else (fn() or True)
            print(f"  {k:14s} {'ok' if ok else 'skipped (缺数据)'}")
        except Exception as e:
            print(f"  {k:14s} 失败: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())