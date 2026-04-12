"""
screener.py
评分、排名与筛选主逻辑

依据 Factor Score 对股票排名，输出 Top N 候选标的。
支持按风险偏好动态调整因子权重。
"""

import logging
import pandas as pd

from .factors import compute_factor_scores, DEFAULT_WEIGHTS, BENCHMARK_WEIGHTS

logger = logging.getLogger(__name__)

# 风险偏好预设权重（在默认基础上微调）
RISK_WEIGHTS = {
    "conservative": {      # 保守：更重视低波动、高品质
        "momentum": 0.20,
        "volatility": 0.35,
        "value": 0.25,
        "quality": 0.20,
    },
    "balanced": DEFAULT_WEIGHTS,  # 平衡：使用默认权重
    "aggressive": {        # 积极：更重视动量
        "momentum": 0.45,
        "volatility": 0.15,
        "value": 0.20,
        "quality": 0.20,
    },
}


def screen_stocks(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_prices: pd.Series | None = None,
    top_n: int = 10,
    risk_profile: str = "balanced",
    custom_weights: dict | None = None,
    min_history: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    执行股票筛选，返回 Top N 排名结果与完整因子分数表。

    Parameters
    ----------
    prices : pd.DataFrame
        复权收盘价宽表。
    fundamentals : pd.DataFrame
        财务数据，index=ticker。
    benchmark_prices : pd.Series | None
        基准指数价格序列；提供时默认启用相对动量优先的 benchmark-aware 选股。
    top_n : int
        输出 Top N 只股票，默认 10。
    risk_profile : str
        风险偏好：'conservative' / 'balanced' / 'aggressive'。
    custom_weights : dict | None
        自定义因子权重，若提供则覆盖 risk_profile。
    min_history : int
        最少交易日数据要求。

    Returns
    -------
    top_n_df : pd.DataFrame
        Top N 股票明细（包含各因子分、综合分、行业信息）。
    all_scores : pd.DataFrame
        全量股票因子评分表（已排序）。
    """
    if custom_weights:
        weights = custom_weights
        logger.info("使用自定义因子权重: %s", weights)
    elif benchmark_prices is not None:
        weights = BENCHMARK_WEIGHTS
        logger.info("启用基准相对动量选股，因子权重: %s", weights)
    else:
        weights = RISK_WEIGHTS.get(risk_profile, DEFAULT_WEIGHTS)
        logger.info("风险偏好=%s，因子权重: %s", risk_profile, weights)

    # 计算因子分
    scores = compute_factor_scores(
        prices=prices,
        fundamentals=fundamentals,
        weights=weights,
        benchmark_prices=benchmark_prices,
        min_history=min_history,
    )

    # 拼接行业信息
    if "sector" in fundamentals.columns:
        scores = scores.join(fundamentals["sector"], how="left")
        scores["sector"] = scores["sector"].fillna("Unknown")

    # 选出 Top N
    top_n_df = scores.head(top_n).copy()
    top_n_df.index.name = "ticker"
    top_n_df = top_n_df.reset_index()

    # 格式化输出列
    display_cols = [
        "ticker", "factor_score",
        "relative_momentum_z",
        "momentum_z", "volatility_z", "value_z", "quality_z",
        "relative_momentum_raw",
        "momentum_raw", "volatility_raw", "value_raw", "quality_raw",
    ]
    if "sector" in top_n_df.columns:
        display_cols.append("sector")

    top_n_df = top_n_df[[c for c in display_cols if c in top_n_df.columns]]

    logger.info("筛选完成，Top %d 股票: %s", top_n, top_n_df["ticker"].tolist())
    return top_n_df, scores
