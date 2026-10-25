"""三张性能图: 线程吞吐 / 解码延迟 vs 序列长度 / 规模 scaling.

数据全部取自 experiments/results/ 已落盘 json, 不新增未实测数据.
产出 docs/figures/:
  perf_threads.png   tok/s 与 ms/step vs 线程数 (按权重规模分档)
  perf_decode.png    解码 ms/step vs 序列长度 (八路径, 含整模型 C 后端)
  perf_scaling.png   栈参数 vs ppl (1.4M / 20M / 100M) + 深度代价律
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
RES = ROOT / "experiments" / "results"

_CJK_OK = False

INK = "#1b1f24"
MUTED = "#6b7280"
ACCENT = "#2f6fb5"
WARM = "#c2410c"
LINE = "#d1d5db"
GREEN = "#2f855a"


def _probe_glyph(font_name: str) -> bool:
    try:
        fig = plt.figure(figsize=(1, 1))
        fig.text(0.1, 0.3, "状态", fontname=font_name, fontsize=24)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return bool((buf < 200).sum() > 60)
    except Exception:
        return False


def _style():
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


def L(zh: str, en: str) -> str:
    return zh if _CJK_OK else en


def _footnote(fig, text: str, y: float = -0.02):
    fig.text(0.5, y, text, ha="center", fontsize=7.8, color=MUTED, wrap=True)


def _load(name: str):
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fig_threads():
    """吞吐与单步耗时 vs 线程数. 权重分档决定最优线程, 这是落地策略的依据."""
    d = _load("_decode_cthreads_size.json")
    if not d:
        return False
    sizes = ["d80", "d128", "d192", "d256", "d384", "d512", "d768"]
    sizes = [s for s in sizes if s in d]
    threads = sorted({int(t) for s in sizes for t in d[s]["ms"]})
    mb = [d[s]["MB"] for s in sizes]

    # 颜色按权重规模深浅分档: 浅色=小模型, 深色=大模型
    cmap = plt.get_cmap("Blues")
    lo, hi = min(mb), max(mb)
    colors = [cmap(0.35 + 0.55 * (m - lo) / max(hi - lo, 1e-9)) for m in mb]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))

    ax = axes[0]
    for s, mbv, c in zip(sizes, mb, colors):
        ms = [d[s]["ms"][str(t)] for t in threads]
        tps = [1000.0 / m for m in ms]
        ax.plot(threads, tps, marker="o", ms=4.6, lw=1.8, color=c,
                label=f"{s} ({mbv:.0f} MB)")
        best_t = d[s]["best"]
        bi = threads.index(best_t) if best_t in threads else None
        if bi is not None:
            ax.scatter([threads[bi]], [tps[bi]], s=64, facecolor="none",
                       edgecolor=WARM, lw=1.5, zorder=6)
    ax.set_xlabel(L("线程数", "threads"))
    ax.set_ylabel(L("解码吞吐 (token/s)", "decode throughput (tok/s)"))
    ax.set_title(L("吞吐随线程数变化（越高越好）", "throughput vs threads (higher better)"),
                 fontsize=10.4)
    ax.set_xticks(threads)
    ax.legend(frameon=False, fontsize=8.2, title=L("权重规模", "weight size"),
              title_fontsize=8.4)

    ax = axes[1]
    for s, mbv, c in zip(sizes, mb, colors):
        ms = [d[s]["ms"][str(t)] for t in threads]
        ax.plot(threads, ms, marker="s", ms=4.4, lw=1.8, color=c,
                label=f"{s} ({mbv:.0f} MB)")
    ax.axvspan(0.7, 1.3, color=GREEN, alpha=0.07)
    ax.axvspan(5.5, 6.5, color=WARM, alpha=0.07)
    ax.set_xlabel(L("线程数", "threads"))
    ax.set_ylabel(L("单步解码耗时 (ms)", "decode ms/step"))
    ax.set_title(L("单步耗时随线程数变化（越低越好）", "ms/step vs threads (lower better)"),
                 fontsize=10.4)
    ax.set_xticks(threads)
    ax.text(1, ax.get_ylim()[1] * 0.96, L("小模型\n单线程最优", "small\n1 thread"),
            ha="center", va="top", fontsize=8, color=GREEN)
    ax.text(6, ax.get_ylim()[1] * 0.96, L("大模型\n物理核最优", "large\nphysical cores"),
            ha="center", va="top", fontsize=8, color=WARM)

    fig.suptitle(L("解码线程策略：最优线程数由权重规模决定（B=1 单序列）",
                   "Decode thread policy: optimal thread count depends on weight size (B=1)"),
                 fontsize=11.5, weight="bold", y=1.02)
    _footnote(fig, L("数据 experiments/results/_decode_cthreads_size.json；每档独立进程测量。"
                     "阈值 32 MB：≤23 MB 单线程、≥39 MB 物理核；圈标为该档实测最优。",
                     "Source _decode_cthreads_size.json; isolated per cell. "
                     "Threshold 32 MB: <=23 MB single thread, >=39 MB physical cores; "
                     "circles mark measured optimum."))
    fig.savefig(FIG / "perf_threads.png")
    plt.close(fig)
    return True


def fig_decode():
    """解码 ms/step vs 序列长度. 八路径含整模型 C 后端, 检验 O(1) vs O(t)."""
    d = _load("_decode_seven.json")
    if not d:
        return False
    Ts = d["Ts"]
    tab = d["table"]
    order = ["sunnyC", "sunny", "transformer", "rwkv", "mamba", "h3",
             "hyena", "linear_attn"]
    names = [n for n in order if n in tab]
    pretty = {"sunnyC": "SunnyNet (C runtime)", "sunny": "SunnyNet (torch)"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    for n in names:
        ys = tab[n]
        if n == "sunnyC":
            st = dict(color=ACCENT, lw=2.8, ms=6.5, zorder=6)
        elif n == "sunny":
            st = dict(color=ACCENT, lw=1.6, ms=4.4, ls="--", alpha=0.75, zorder=5)
        elif n == "transformer":
            st = dict(color=WARM, lw=2.2, ms=5.4, zorder=4)
        elif n == "rwkv":
            st = dict(color="#7c3aed", lw=1.8, ms=4.4, zorder=3)
        else:
            st = dict(color=MUTED, lw=1.1, ms=3.4, alpha=0.8)
        ax.plot(Ts, ys, marker="o", label=pretty.get(n, n), **st)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel(L("序列长度 T", "sequence length T"))
    ax.set_ylabel(L("单步解码耗时 (ms)", "decode ms/step"))
    ax.set_title(L("八路径解码延迟（B=1 单序列）", "decode latency, eight paths (B=1)"),
                 fontsize=10.4)
    ax.legend(frameon=False, ncol=2, fontsize=8.2)

    ax = axes[1]
    tf = tab["transformer"]
    flat = tab["sunnyC"]
    ax.plot(Ts, tf, marker="o", ms=5.4, lw=2.2, color=WARM, label="Transformer (O(t))")
    ax.plot(Ts, flat, marker="o", ms=6.0, lw=2.8, color=ACCENT,
            label="SunnyNet C runtime (O(1))")
    for t, v in zip(Ts, flat):
        ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=ACCENT)
    for t, v in zip(Ts, tf):
        ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=WARM)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel(L("序列长度 T", "sequence length T"))
    ax.set_ylabel(L("单步解码耗时 (ms)", "decode ms/step"))
    ax.set_title(L("平坦 vs 线性增长：优势来自架构",
                   "flat vs linear: the advantage is architectural"),
                 fontsize=10.4)
    ax.legend(frameon=False, fontsize=9)
    ratio = tf[-1] / flat[-1]
    ax.annotate(L(f"T=4096 快 {ratio:.2f}×", f"{ratio:.2f}x at T=4096"),
                xy=(Ts[-1], flat[-1]), xytext=(Ts[-3], 2.95),
                fontsize=9.2, color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=LINE, lw=0.9),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))

    fig.suptitle(L("解码延迟随序列长度：线性递推保持常数代价",
                   "Decode latency vs sequence length: the recurrence stays constant"),
                 fontsize=11.6, weight="bold", y=1.02)
    _footnote(fig, L(f"数据 experiments/results/_decode_seven.json（{d.get('rounds','?')} 轮交错）。"
                     "sunnyC 为整模型 C 后端单步；sunny/torch 路径单独保留。",
                     f"Source _decode_seven.json ({d.get('rounds','?')} interleaved rounds). "
                     "sunnyC is the whole-model C backend single step; the torch path is kept separate."))
    fig.savefig(FIG / "perf_decode.png")
    plt.close(fig)
    return True


# 栈参数对齐的 ppl 点 (预设, 栈参数, ppl) —— 取自 report_scale / oncompare / sc_* 落盘值
SCALE_POINTS = [
    ("1.4M", 365332, 600, 57.83),
    ("20M", 16761394, 300, 241.84),
    ("100M", 93828904, 100, 109.47),
]

# 深度代价律: 固定 dim=128, T=256, B=8 的层数扫描 (§28 实测)
DEPTH_L = [1, 2, 3, 4, 6, 8, 10, 12]
DEPTH_MS = [121.4, 158.9, 195.6, 267.8, 310.2, 365.8, 414.2, 499.5]


def fig_scaling():
    """规模 scaling: 栈参数 vs ppl, 以及深度代价律. 两者都标注口径差异."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # 左: 栈参数 vs ppl (zipf 风格, 横轴对数)
    ax = axes[0]
    xs = [p[1] for p in SCALE_POINTS]
    ys = [p[3] for p in SCALE_POINTS]
    labels = [p[0] for p in SCALE_POINTS]
    ax.plot(xs, ys, marker="o", ms=7.5, lw=2.0, color=ACCENT, zorder=5)
    for x, y, lab, (_, _, steps, _) in zip(xs, ys, labels, SCALE_POINTS):
        ax.annotate(f"{lab}\nppl {y:.1f} / {steps} 步", (x, y),
                    textcoords="offset points", xytext=(4, 13),
                    ha="left", fontsize=8.6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(xs[0] * 0.35, xs[-1] * 3.0)
    ax.set_xlabel(L("计算栈参数（对数轴）", "stack parameters (log)"))
    ax.set_ylabel(L("验证困惑度 ppl（越低越好）", "val perplexity (lower better)"))
    ax.set_title(L("质量随规模：非单调", "quality vs scale: not monotonic"),
                 fontsize=10.4)
    ax.set_ylim(min(ys) * 0.4, max(ys) * 1.35)
    ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[2], ys[2]),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6,
                                connectionstyle="arc3,rad=0.22"))
    ax.text(0.97, 0.60,
            L("100M 回升，但步数档不同\n（100 步 vs 300 步）",
              "100M improves, but the step\nbudgets differ (100 vs 300)"),
            transform=ax.transAxes, fontsize=8, color=GREEN,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, lw=0.8, alpha=0.9))

    # 右: 深度代价律
    ax = axes[1]
    ax.plot(DEPTH_L, DEPTH_MS, marker="o", ms=5.4, lw=2.0, color=ACCENT,
            label=L("实测 ms/step", "measured ms/step"))
    coef = np.polyfit(DEPTH_L, DEPTH_MS, 1)
    fit = np.poly1d(coef)
    ax.plot(DEPTH_L, fit(DEPTH_L), ls="--", lw=1.5, color=WARM,
            label=L(f"最小二乘 {coef[0]:.1f}·L + {coef[1]:.1f}",
                    f"least squares {coef[0]:.1f}L + {coef[1]:.1f}"))
    for i, (x, y) in enumerate(zip(DEPTH_L, DEPTH_MS)):
        dx, dy = (16, -12) if i == 0 else (( -18, 2) if i == 6 else (0, 9))
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=7.8, color=INK,
                    clip_on=False)
    ax.set_xlabel(L("层数 L", "layers L"))
    ax.set_ylabel(L("训练步耗时 (ms)", "training step (ms)"))
    ax.set_title(L("深度代价是线性的", "depth cost is linear"), fontsize=10.4)
    ax.set_ylim(min(DEPTH_MS) * 0.80, max(DEPTH_MS) * 1.08)
    ax.set_xlim(min(DEPTH_L) - 0.9, max(DEPTH_L) + 0.9)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    fig.suptitle(L("规模与深度：两条实测口径", "Scale and depth: two measured views"),
                 fontsize=11.6, weight="bold", y=1.02)
    _footnote(fig, L("左图数据 report_scale.json / oncompare_*.json / sc_100m_*.json，"
                     "三档步数不同（600/300/100），只读趋势，禁跨档相减；"
                     "右图 d=128、T=256、B=8、6 线程、3 轮交错独立子进程。",
                     "Left: report_scale.json / oncompare_*.json / sc_100m_*.json; "
                     "step budgets differ (600/300/100), read the trend only. "
                     "Right: d=128, T=256, B=8, 6 threads, 3 interleaved isolated rounds."))
    fig.savefig(FIG / "perf_scaling.png")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    _style()
    print(f"  CJK glyphs available: {_CJK_OK}")
    todo = {"threads": fig_threads, "decode": fig_decode, "scaling": fig_scaling}
    for k, fn in todo.items():
        if args.only and args.only != k:
            continue
        try:
            ok = fn()
            print(f"  {k:10s} {'ok' if ok else 'skipped (no data)'}")
        except Exception as e:
            print(f"  {k:10s} FAILED: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())