"""
visualizer.py
图表生成模块

提供：
  1. Factor Score 排名条形图（Top 20）
  2. 组合持仓权重饼图
  3. 回测净值曲线（含 Baseline 对比线）
  4. 各因子分布箱线图
  5. 单支股票四维雷达图
"""

import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#1D3557",
    "accent": "#E63946",
    "green": "#2DC653",
    "gray": "#8D99AE",
    "light": "#F1FAEE",
    "baseline": "#457B9D",
}


def _factor_specs(scores: pd.DataFrame) -> list[tuple[str, str, str]]:
    """返回当前数据中存在的因子列、显示名和颜色。"""
    specs = [
        ("relative_momentum_z", "相对动量 Relative Momentum", PALETTE["primary"]),
        ("momentum_z", "动量 Momentum", PALETTE["accent"]),
        ("volatility_z", "波动 Volatility", PALETTE["baseline"]),
        ("value_z", "估值 Value", PALETTE["gray"]),
        ("quality_z", "品质 Quality", PALETTE["green"]),
    ]
    return [spec for spec in specs if spec[0] in scores.columns]


def _save(fig: plt.Figure, name: str, show: bool = False) -> str:
    path = str(OUTPUT_DIR / f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("图表已保存: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Factor Score 排名条形图
# ---------------------------------------------------------------------------

def plot_factor_scores(scores: pd.DataFrame, top_n: int = 20, show: bool = False) -> str:
    """
    绘制 Top N 股票 Factor Score 水平条形图。

    Parameters
    ----------
    scores : pd.DataFrame
        全量因子评分表（含 factor_score 列），已降序排列。
    top_n : int
        显示 Top N。
    show : bool
        是否在屏幕显示。

    Returns
    -------
    str
        保存路径。
    """
    df = scores.head(top_n).copy()
    if "ticker" in df.columns:
        df = df.set_index("ticker")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    colors = [PALETTE["accent"] if i < 10 else PALETTE["gray"] for i in range(len(df))]
    bars = ax.barh(df.index[::-1], df["factor_score"][::-1], color=colors[::-1], height=0.7)

    ax.set_xlabel("Factor Score (Z-score 加权综合分)", fontsize=11)
    ax.set_title(f"Top {top_n} 股票 Factor Score 排名", fontsize=14, fontweight="bold", pad=15)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, df["factor_score"][::-1]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", fontsize=9,
        )

    fig.tight_layout()
    return _save(fig, "factor_score_ranking", show)


# ---------------------------------------------------------------------------
# 2. 持仓权重饼图
# ---------------------------------------------------------------------------

def plot_portfolio_weights(portfolio: pd.DataFrame, show: bool = False) -> str:
    """
    绘制组合持仓权重饼图。

    Parameters
    ----------
    portfolio : pd.DataFrame
        columns: [ticker, weight_pct, ...]
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = portfolio["ticker"].tolist()
    sizes = portfolio["weight_pct"].tolist()

    cmap = plt.cm.get_cmap("tab20", len(labels))
    colors = [cmap(i) for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.85, labeldistance=1.05,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title("组合持仓权重分布", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return _save(fig, "portfolio_weights", show)


# ---------------------------------------------------------------------------
# 3. 回测净值曲线
# ---------------------------------------------------------------------------

def plot_nav_curve(
    strategy_nav: pd.Series,
    baseline_nav: pd.Series,
    metrics: dict,
    show: bool = False,
) -> str:
    """
    绘制回测净值曲线（策略 vs. Baseline）。
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax1, ax2 = axes

    # 净值曲线
    ax1.plot(strategy_nav.index, strategy_nav.values,
             color=PALETTE["accent"], linewidth=2, label="策略组合")
    ax1.plot(baseline_nav.index, baseline_nav.values,
             color=PALETTE["baseline"], linewidth=1.5, linestyle="--", label="S&P 500 Baseline")
    ax1.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax1.fill_between(strategy_nav.index, 1, strategy_nav.values,
                     where=(strategy_nav.values >= 1),
                     alpha=0.1, color=PALETTE["green"])
    ax1.fill_between(strategy_nav.index, 1, strategy_nav.values,
                     where=(strategy_nav.values < 1),
                     alpha=0.1, color=PALETTE["accent"])
    ax1.set_ylabel("净值", fontsize=11)
    ax1.set_title("回测净值曲线", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)

    # 绩效标注
    s = metrics
    text = (
        f"策略  CAGR={s['strategy_cagr']:.1%}  Sharpe={s['strategy_sharpe']:.2f}  MDD={s['strategy_max_drawdown']:.1%}\n"
        f"基准  CAGR={s['baseline_cagr']:.1%}  Sharpe={s['baseline_sharpe']:.2f}  MDD={s['baseline_max_drawdown']:.1%}\n"
        f"IR={s['information_ratio']:.2f}  月胜率={s['monthly_win_rate']:.1%}"
    )
    ax1.text(
        0.01, 0.97, text, transform=ax1.transAxes,
        va="top", fontsize=9, family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8, "edgecolor": "#ccc"},
    )

    # 相对超额收益
    aligned = strategy_nav.reindex(baseline_nav.index)
    relative = (aligned / baseline_nav - 1) * 100
    colors_rel = [PALETTE["green"] if v >= 0 else PALETTE["accent"] for v in relative.values]
    ax2.bar(relative.index, relative.values, color=colors_rel, width=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("相对超额 (%)", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "nav_curve", show)


# ---------------------------------------------------------------------------
# 4. 因子分布箱线图
# ---------------------------------------------------------------------------

def plot_factor_distribution(scores: pd.DataFrame, show: bool = False) -> str:
    """
    绘制当前可用因子标准化分数分布箱线图。
    """
    factor_specs = _factor_specs(scores)
    if not factor_specs:
        raise ValueError("scores 中没有可用于绘图的因子列。")

    fig, axes = plt.subplots(1, len(factor_specs), figsize=(max(8, 3.2 * len(factor_specs)), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (col, name, color) in zip(axes, factor_specs):
        data = scores[col].dropna()
        bp = ax.boxplot(
            data, vert=True, patch_artist=True,
            widths=0.5,
            boxprops={"facecolor": color, "alpha": 0.6},
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4, "color": color},
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Z-score" if ax == axes[0] else "")

    fig.suptitle("因子 Z-score 分布", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "factor_distribution", show)


# ---------------------------------------------------------------------------
# 5. 雷达图（单支股票四维因子对比）
# ---------------------------------------------------------------------------

def plot_radar(top_n_df: pd.DataFrame, show: bool = False) -> str:
    """
    绘制 Top N 股票四维因子雷达图。

    Parameters
    ----------
    top_n_df : pd.DataFrame
        Top N 股票明细，含因子 z-score 列。
    """
    factor_specs = _factor_specs(top_n_df)
    if not factor_specs:
        raise ValueError("top_n_df 中没有可用于雷达图的因子列。")

    factor_cols = [col for col, _, _ in factor_specs]
    labels = [name for _, name, _ in factor_specs]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    df = top_n_df.set_index("ticker") if "ticker" in top_n_df.columns else top_n_df
    # 限制最多 10 只（避免图表过于密集）
    df = df.head(10)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    cmap = plt.cm.get_cmap("tab10", len(df))

    for i, (ticker, row) in enumerate(df.iterrows()):
        values = [row.get(c, 0) for c in factor_cols]
        # 缩放至 0~1 范围（基于当前样本）
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, linewidth=1.5, color=cmap(i), label=ticker)
        ax.fill(angles, values_plot, alpha=0.1, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title("Top N 股票四维因子雷达图", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    return _save(fig, "radar_chart", show)
