"""
backtester.py
回测引擎

严格遵循时间序列切割，防止数据泄漏：
  - In-Sample  : 2015-01-01 ~ 2022-12-31
  - Out-of-Sample: 2023-01-01 ~ 2024-12-31

支持：
  - 月度 / 季度 / 半年度 再平衡
    - Top N 策略 vs. S&P 500 指数 Baseline 对比
  - 输出 CAGR、Sharpe、最大回撤、信息比率、月度胜率
"""

import logging
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REBALANCE_FREQ_MAP = {
    "monthly": "MS",
    "quarterly": "QS",
    "semi-annual": "6MS",
}


# ---------------------------------------------------------------------------
# 回测核心
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_prices: pd.Series,
    start: str,
    end: str,
    top_n: int = 10,
    risk_profile: str = "balanced",
    rebalance_freq: str = "monthly",
    weight_method: str = "equal_weight",
    min_history_days: int = 252,
    risk_free_rate: float = 0.05,
    transaction_cost: float = 0.001,
) -> dict:
    """
    运行完整回测，返回净值曲线与绩效指标。

    Parameters
    ----------
    prices : pd.DataFrame
        全区间复权收盘价宽表。
    fundamentals : pd.DataFrame
        财务数据（point-in-time 近似：使用当期快照）。
    benchmark_prices : pd.Series
        基准指数复权收盘价序列（如 S&P 500 的 '^GSPC'）。
    start, end : str
        回测期间（YYYY-MM-DD）。
    top_n : int
        持仓数量。
    risk_profile : str
        风险偏好。
    rebalance_freq : str
        'monthly' / 'quarterly' / 'semi-annual'
    weight_method : str
        'equal_weight' / 'risk_parity'
    min_history_days : int
        有效数据最小交易日。
    risk_free_rate : float
        年化无风险利率（用于 Sharpe）。
    transaction_cost : float
        单边交易成本比例，默认 0.1%。

    Returns
    -------
    dict
        keys: strategy_nav, baseline_nav, metrics, rebalance_dates, portfolio_history
    """
    # 延迟导入避免循环依赖
    from .screener import screen_stocks
    from .portfolio import equal_weight, risk_parity

    prices_bt = prices.loc[start:end].copy()
    if prices_bt.empty:
        raise ValueError(f"回测期间 {start}~{end} 无价格数据。")

    benchmark_bt = benchmark_prices.loc[start:end].dropna().copy()
    if benchmark_bt.empty:
        raise ValueError(f"回测期间 {start}~{end} 无基准指数价格数据。")

    baseline_ret_full = benchmark_bt.pct_change().dropna()

    freq_str = REBALANCE_FREQ_MAP.get(rebalance_freq, "MS")
    rebalance_dates = pd.date_range(
        start=prices_bt.index[0],
        end=prices_bt.index[-1],
        freq=freq_str,
    )
    rebalance_dates = rebalance_dates[rebalance_dates.isin(prices_bt.index)]
    if len(rebalance_dates) == 0:
        # 回退到最近的交易日
        rebalance_dates = pd.DatetimeIndex([prices_bt.index[0]])

    logger.info(
        "回测配置：%s ~ %s，再平衡 %d 次（%s），Top %d，权重=%s",
        start, end, len(rebalance_dates), rebalance_freq, top_n, weight_method,
    )

    strategy_returns = []
    portfolio_history = []
    prev_weights = None

    for i, rb_date in enumerate(rebalance_dates):
        # 下一次再平衡日（或期末）
        next_rb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else prices_bt.index[-1]

        # 使用 rb_date 前的历史数据做因子计算（防数据泄漏）
        hist_end = rb_date
        hist_start_min = (pd.Timestamp(hist_end) - relativedelta(years=2)).strftime("%Y-%m-%d")
        prices_hist = prices.loc[hist_start_min:hist_end]
        benchmark_hist = benchmark_prices.loc[hist_start_min:hist_end]

        try:
            top_n_df, _ = screen_stocks(
                prices=prices_hist,
                fundamentals=fundamentals,
                benchmark_prices=benchmark_hist,
                top_n=top_n,
                risk_profile=risk_profile,
                min_history=min_history_days,
            )
            selected_tickers = top_n_df["ticker"].tolist()
        except Exception as e:
            logger.warning("  [%s] 因子计算失败 (%s)，跳过本期", rb_date.date(), e)
            continue

        if not selected_tickers:
            logger.warning("  [%s] 无有效股票，跳过本期", rb_date.date())
            continue

        # 持仓期间收益
        period_prices = prices_bt.loc[rb_date:next_rb]
        if len(period_prices) < 2:
            continue

        # 策略组合收益
        available = [t for t in selected_tickers if t in period_prices.columns]
        if not available:
            continue

        if weight_method == "risk_parity":
            w = risk_parity(available, prices_hist)
        else:
            w = equal_weight(available)

        w = w.reindex(available).fillna(0)
        if w.sum() > 0:
            w = w / w.sum()

        period_ret = period_prices[available].pct_change().fillna(0)
        strategy_period_ret = (period_ret * w).sum(axis=1)

        # 将交易成本计入再平衡后的首个可交易日收益，避免无摩擦回测过于乐观。
        if prev_weights is None:
            cost_rate = transaction_cost
        else:
            all_tickers = prev_weights.index.union(w.index)
            prev_w = prev_weights.reindex(all_tickers, fill_value=0.0)
            curr_w = w.reindex(all_tickers, fill_value=0.0)
            turnover = (curr_w - prev_w).abs().sum() / 2
            cost_rate = turnover * 2 * transaction_cost

        if len(strategy_period_ret) > 1 and cost_rate > 0:
            strategy_period_ret.iloc[1] -= cost_rate

        strategy_returns.append(strategy_period_ret.iloc[1:])  # 去掉第一行 0
        prev_weights = w.copy()

        portfolio_history.append({
            "date": rb_date,
            "tickers": available,
            "weights": w.to_dict(),
        })

    if not strategy_returns:
        raise RuntimeError("回测无有效期间，请检查数据与参数。")

    # 合并每日收益序列
    strat_ret = pd.concat(strategy_returns).sort_index()
    base_ret = baseline_ret_full.sort_index()

    # 去重（相邻再平衡日重叠部分）
    strat_ret = strat_ret[~strat_ret.index.duplicated(keep="last")]
    base_ret = base_ret[~base_ret.index.duplicated(keep="last")]

    # 统一对齐日期，确保策略与基准在相同交易日上比较
    common_idx = strat_ret.index.intersection(base_ret.index)
    if common_idx.empty:
        raise RuntimeError("策略与基准收益序列无重叠交易日，无法评估。")
    strat_ret = strat_ret.loc[common_idx]
    base_ret = base_ret.loc[common_idx]

    # 净值曲线
    strategy_nav = (1 + strat_ret).cumprod()
    baseline_nav = (1 + base_ret).cumprod()
    strategy_nav.name = "Strategy"
    baseline_nav.name = "Baseline"

    # 绩效指标
    metrics = _calc_metrics(strat_ret, base_ret, risk_free_rate)

    return {
        "strategy_nav": strategy_nav,
        "baseline_nav": baseline_nav,
        "strategy_returns": strat_ret,
        "baseline_returns": base_ret,
        "metrics": metrics,
        "rebalance_dates": rebalance_dates,
        "portfolio_history": portfolio_history,
    }


# ---------------------------------------------------------------------------
# 绩效指标
# ---------------------------------------------------------------------------

def _calc_metrics(
    strategy_ret: pd.Series,
    baseline_ret: pd.Series,
    risk_free_rate: float = 0.05,
) -> dict:
    """计算年化绩效指标。"""
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

    def _sharpe(ret):
        excess = ret - rf_daily
        return (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    def _cagr(ret):
        n_years = len(ret) / 252
        total = (1 + ret).prod()
        return total ** (1 / n_years) - 1 if n_years > 0 else 0.0

    def _max_drawdown(ret):
        nav = (1 + ret).cumprod()
        roll_max = nav.cummax()
        dd = (nav - roll_max) / roll_max
        return dd.min()

    # 月度收益（用于胜率）
    def _monthly_returns(ret):
        return (1 + ret).resample("ME").prod() - 1

    s_cagr = _cagr(strategy_ret)
    b_cagr = _cagr(baseline_ret)
    s_sharpe = _sharpe(strategy_ret)
    b_sharpe = _sharpe(baseline_ret)
    s_mdd = _max_drawdown(strategy_ret)
    b_mdd = _max_drawdown(baseline_ret)

    # 信息比率
    excess = strategy_ret - baseline_ret
    ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0.0

    # 月度胜率
    s_monthly = _monthly_returns(strategy_ret)
    b_monthly = _monthly_returns(baseline_ret)
    common_months = s_monthly.index.intersection(b_monthly.index)
    if len(common_months) > 0:
        win_rate = (s_monthly[common_months] > b_monthly[common_months]).mean()
    else:
        win_rate = np.nan

    metrics = {
        "strategy_cagr": s_cagr,
        "baseline_cagr": b_cagr,
        "strategy_sharpe": s_sharpe,
        "baseline_sharpe": b_sharpe,
        "strategy_max_drawdown": s_mdd,
        "baseline_max_drawdown": b_mdd,
        "information_ratio": ir,
        "monthly_win_rate": win_rate,
        "strategy_annualized_vol": strategy_ret.std() * np.sqrt(252),
        "baseline_annualized_vol": baseline_ret.std() * np.sqrt(252),
    }

    logger.info(
        "回测指标 | 策略 CAGR=%.2f%% Sharpe=%.2f MDD=%.1f%% | Baseline CAGR=%.2f%% Sharpe=%.2f MDD=%.1f%% | IR=%.2f 月胜率=%.1f%%",
        s_cagr * 100, s_sharpe, s_mdd * 100,
        b_cagr * 100, b_sharpe, b_mdd * 100,
        ir, win_rate * 100 if not np.isnan(win_rate) else 0,
    )
    return metrics
