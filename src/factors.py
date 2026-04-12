"""
factors.py
四大因子计算模块

因子维度：
  1. 动量 Momentum  (30%) —— 12-1 月动量，越高越好
  2. 波动 Volatility (25%) —— 年化波动率，越低越好
  3. 估值 Value      (25%) —— 行业相对 P/E & P/B，越低越好
  4. 品质 Quality    (20%) —— ROE + EPS 稳定性，越高越好
    5. 相对动量 Relative Momentum —— 相对 S&P 500 的 12-1 月超额动量，越高越好

所有因子均经过：
  - Winsorize（1%–99% 截尾）
  - Z-score 标准化
  - 方向对齐（低分因子取负，统一为"高分=好"）
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import mstats

logger = logging.getLogger(__name__)

# 默认因子权重
DEFAULT_WEIGHTS = {
    "momentum": 0.30,
    "volatility": 0.25,
    "value": 0.25,
    "quality": 0.20,
}

# 当提供基准指数时，使用更稳健的相对动量 + 风险/基本面混合信号
BENCHMARK_WEIGHTS = {
    "relative_momentum": 0.50,
    "volatility": 0.15,
    "value": 0.15,
    "quality": 0.20,
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def winsorize(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """对 Series 进行 Winsorize 截尾处理。"""
    arr = mstats.winsorize(series.dropna(), limits=limits)
    result = pd.Series(arr, index=series.dropna().index)
    return result.reindex(series.index)


def zscore(series: pd.Series) -> pd.Series:
    """Z-score 标准化，忽略 NaN。"""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def calc_momentum_series(price_series: pd.Series, lookback: int = 252, skip: int = 21) -> float:
    """计算单条价格序列的 12-1 月动量。"""
    if len(price_series) < lookback + skip:
        return np.nan

    past = price_series.iloc[-(lookback + skip)]
    skip_price = price_series.iloc[-skip]
    if pd.isna(past) or pd.isna(skip_price) or past == 0:
        return np.nan

    return float(skip_price / past - 1)


def normalize_factor(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Winsorize → Z-score → 方向对齐。

    Parameters
    ----------
    series : pd.Series
    higher_is_better : bool
        若 False（如波动率），取反使得"高分=好"。
    """
    s = winsorize(series)
    s = zscore(s)
    if not higher_is_better:
        s = -s
    return s


# ---------------------------------------------------------------------------
# 因子 1：动量 Momentum
# ---------------------------------------------------------------------------

def calc_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """
    计算 12-1 月动量（过去 12 个月收益率，剔除最近 1 个月）。

    Parameters
    ----------
    prices : pd.DataFrame
        复权收盘价宽表（行=日期，列=ticker）。
    lookback : int
        动量回望窗口（交易日），默认 252（约 12 个月）。
    skip : int
        跳过最近 N 个交易日（默认 21，约 1 个月）。

    Returns
    -------
    pd.Series
        各股票动量原始值，index=ticker。
    """
    latest = prices.iloc[-1]
    past = prices.iloc[-(lookback + skip)]
    skip_price = prices.iloc[-skip]

    # 12-1 月动量 = (t-1月价格 / t-12月价格) - 1
    momentum = skip_price / past - 1
    momentum = momentum.replace([np.inf, -np.inf], np.nan)
    logger.debug("动量因子计算完成，有效值 %d 只", momentum.notna().sum())
    return momentum


def calc_relative_momentum(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """
    计算相对动量：个股 12-1 月动量减去基准指数 12-1 月动量。
    """
    stock_momentum = calc_momentum(prices, lookback=lookback, skip=skip)
    benchmark_momentum = calc_momentum_series(benchmark_prices, lookback=lookback, skip=skip)

    if pd.isna(benchmark_momentum):
        logger.warning("基准动量不可用，退回使用绝对动量作为相对动量代理")
        return stock_momentum

    relative_momentum = stock_momentum - benchmark_momentum
    relative_momentum = relative_momentum.replace([np.inf, -np.inf], np.nan)
    logger.debug("相对动量计算完成，有效值 %d 只", relative_momentum.notna().sum())
    return relative_momentum


# ---------------------------------------------------------------------------
# 因子 2：波动率 Volatility
# ---------------------------------------------------------------------------

def calc_volatility(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    计算年化波动率（过去 window 个交易日日收益率标准差 × √252）。

    Returns
    -------
    pd.Series
        年化波动率，越低越好。
    """
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        logger.warning("价格数据不足 %d 行，使用全部数据计算波动率", window)
        vol = returns.std() * np.sqrt(252)
    else:
        vol = returns.iloc[-window:].std() * np.sqrt(252)

    vol = vol.replace([np.inf, -np.inf], np.nan)
    logger.debug("波动率因子计算完成，有效值 %d 只", vol.notna().sum())
    return vol


# ---------------------------------------------------------------------------
# 因子 3：估值 Value
# ---------------------------------------------------------------------------

def calc_value(
    fundamentals: pd.DataFrame,
    tickers: list[str],
) -> pd.Series:
    """
    计算行业相对 P/E 与 P/B 综合排名（行业内百分位均值）。

    Parameters
    ----------
    fundamentals : pd.DataFrame
        财务数据，index=ticker，需含 pe_ratio、pb_ratio、sector 列。
    tickers : list[str]
        当前有效股票池。

    Returns
    -------
    pd.Series
        估值因子原始值（越低越好，后续取反标准化）。
    """
    fund = fundamentals.loc[fundamentals.index.isin(tickers)].copy()

    def industry_rank(col: str) -> pd.Series:
        """行业内百分位排名（0–1，越低估值越低）。"""
        result = pd.Series(np.nan, index=fund.index)
        for sector, group in fund.groupby("sector"):
            valid = group[col].dropna()
            if len(valid) == 0:
                continue
            ranks = valid.rank(pct=True)
            result.loc[ranks.index] = ranks
        return result

    pe_rank = industry_rank("pe_ratio")
    pb_rank = industry_rank("pb_ratio")

    # 综合估值分 = (PE 百分位 + PB 百分位) / 2，越低代表估值越低（越好）
    value_score = (pe_rank + pb_rank) / 2
    value_score = value_score.replace([np.inf, -np.inf], np.nan)
    logger.debug("估值因子计算完成，有效值 %d 只", value_score.notna().sum())
    return value_score


# ---------------------------------------------------------------------------
# 因子 4：品质 Quality
# ---------------------------------------------------------------------------

def calc_quality(fundamentals: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    计算品质因子：ROE + EPS 稳定性（ROE 越高、盈利越稳定越好）。

    当前版本使用 yfinance 中的 ROE（returnOnEquity）作为主要指标。
    EPS 稳定性简化为 ROE 绝对值（数据限制）。

    Returns
    -------
    pd.Series
        品质因子原始值，越高越好。
    """
    fund = fundamentals.loc[fundamentals.index.isin(tickers)].copy()
    roe = fund["roe"].clip(lower=-1, upper=2)  # 合理范围截尾
    quality = roe.fillna(roe.median())
    logger.debug("品质因子计算完成，有效值 %d 只", quality.notna().sum())
    return quality


# ---------------------------------------------------------------------------
# 综合因子评分
# ---------------------------------------------------------------------------

def compute_factor_scores(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights: dict | None = None,
    benchmark_prices: pd.Series | None = None,
    min_history: int = 252,
) -> pd.DataFrame:
    """
    计算四大因子标准化分数与综合 Factor Score。

    Parameters
    ----------
    prices : pd.DataFrame
        复权收盘价宽表。
    fundamentals : pd.DataFrame
        财务数据。
    weights : dict | None
        因子权重字典，默认使用 DEFAULT_WEIGHTS。
    benchmark_prices : pd.Series | None
        基准指数价格序列，提供时会计算相对动量。
    min_history : int
        最少交易日数据要求，不足则剔除。

    Returns
    -------
    pd.DataFrame
        columns: [momentum_raw, volatility_raw, value_raw, quality_raw,
                  relative_momentum_raw,
                  momentum_z, volatility_z, value_z, quality_z, relative_momentum_z, factor_score]
        index: ticker
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    benchmark_momentum = np.nan
    benchmark_prices_slice = None
    if benchmark_prices is not None:
        benchmark_prices_slice = benchmark_prices.loc[prices.index.min():prices.index.max()].dropna()
        benchmark_momentum = calc_momentum_series(benchmark_prices_slice)

    # 筛选数据充足的股票
    valid_count = prices.notna().sum()
    valid_tickers = valid_count[valid_count >= min_history].index.tolist()
    prices_valid = prices[valid_tickers]
    logger.info("有效股票数（≥%d 交易日）: %d", min_history, len(valid_tickers))

    # 计算原始因子
    mom_raw = calc_momentum(prices_valid)
    vol_raw = calc_volatility(prices_valid)
    val_raw = calc_value(fundamentals, valid_tickers)
    qlt_raw = calc_quality(fundamentals, valid_tickers)
    rel_mom_raw = mom_raw - benchmark_momentum if pd.notna(benchmark_momentum) else mom_raw

    # 对齐 index
    all_tickers = list(set(mom_raw.index) | set(vol_raw.index) |
                       set(val_raw.index) | set(qlt_raw.index))

    df = pd.DataFrame(index=all_tickers)
    df["momentum_raw"] = mom_raw
    df["volatility_raw"] = vol_raw
    df["value_raw"] = val_raw
    df["quality_raw"] = qlt_raw
    df["relative_momentum_raw"] = rel_mom_raw

    # 标准化（方向对齐）
    df["momentum_z"] = normalize_factor(df["momentum_raw"], higher_is_better=True)
    df["volatility_z"] = normalize_factor(df["volatility_raw"], higher_is_better=False)
    df["value_z"] = normalize_factor(df["value_raw"], higher_is_better=False)
    df["quality_z"] = normalize_factor(df["quality_raw"], higher_is_better=True)
    df["relative_momentum_z"] = normalize_factor(df["relative_momentum_raw"], higher_is_better=True)

    # 综合 Factor Score
    score = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for factor_name, weight in weights.items():
        z_col = f"{factor_name}_z"
        if z_col not in df.columns:
            continue
        score = score + weight * df[z_col].fillna(0)
        total_weight += weight

    if total_weight > 0:
        score = score / total_weight

    df["factor_score"] = score

    df = df.sort_values("factor_score", ascending=False)
    logger.info("因子评分计算完成，共 %d 只股票", len(df))
    return df
