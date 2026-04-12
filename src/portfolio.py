"""
portfolio.py
组合构建与权重计算模块

支持：
  1. 等权重（Equal Weight）
  2. 风险平价（Risk Parity）—— 按波动率倒数分配权重
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight(tickers: list[str]) -> pd.Series:
    """
    等权重分配。

    Returns
    -------
    pd.Series
        index=ticker，values=权重（和为 1）。
    """
    n = len(tickers)
    weights = pd.Series(1.0 / n, index=tickers)
    logger.info("等权重组合：%d 只股票，每只 %.2f%%", n, 100 / n)
    return weights


def risk_parity(tickers: list[str], prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    风险平价权重：按波动率倒数分配，再归一化。

    Parameters
    ----------
    tickers : list[str]
        目标股票代码列表。
    prices : pd.DataFrame
        复权收盘价宽表。
    window : int
        波动率计算窗口（交易日）。

    Returns
    -------
    pd.Series
        index=ticker，values=权重（和为 1）。
    """
    available = [t for t in tickers if t in prices.columns]
    if len(available) == 0:
        raise ValueError("tickers 中没有任何股票存在于价格数据中。")

    returns = prices[available].pct_change().dropna()
    if len(returns) < window:
        vol = returns.std() * np.sqrt(252)
    else:
        vol = returns.iloc[-window:].std() * np.sqrt(252)

    vol = vol.replace(0, np.nan).dropna()
    if vol.empty:
        logger.warning("所有股票波动率为 0 或 NaN，退回等权重")
        return equal_weight(tickers)

    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    # 补全缺失（数据不足的股票给 0）
    weights = weights.reindex(tickers, fill_value=0.0)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    logger.info(
        "风险平价组合：%d 只股票，最大权重 %.2f%%，最小权重 %.2f%%",
        len(tickers), weights.max() * 100, weights.min() * 100,
    )
    return weights


def build_portfolio(
    top_n_df: pd.DataFrame,
    prices: pd.DataFrame,
    method: str = "equal_weight",
    capital: float = 100_000.0,
) -> pd.DataFrame:
    """
    根据筛选结果构建组合，计算持仓权重与建议金额。

    Parameters
    ----------
    top_n_df : pd.DataFrame
        Top N 股票明细，需含 ticker 列。
    prices : pd.DataFrame
        复权收盘价宽表（用于风险平价）。
    method : str
        权重方案：'equal_weight' 或 'risk_parity'。
    capital : float
        投资本金（美元），默认 100,000。

    Returns
    -------
    pd.DataFrame
        columns: [ticker, weight_pct, amount_usd, latest_price, shares_approx]
    """
    tickers = top_n_df["ticker"].tolist()

    if method == "risk_parity":
        weights = risk_parity(tickers, prices)
    else:
        weights = equal_weight(tickers)

    # 最新价格
    latest_prices = prices[tickers].iloc[-1] if all(t in prices.columns for t in tickers) else pd.Series(np.nan, index=tickers)

    portfolio = pd.DataFrame({"ticker": tickers})
    portfolio["weight_pct"] = portfolio["ticker"].map(weights) * 100
    portfolio["amount_usd"] = portfolio["weight_pct"] / 100 * capital
    portfolio["latest_price"] = portfolio["ticker"].map(latest_prices)
    portfolio["shares_approx"] = (
        portfolio["amount_usd"] / portfolio["latest_price"]
    ).round(2)

    portfolio = portfolio.sort_values("weight_pct", ascending=False).reset_index(drop=True)
    logger.info(
        "组合构建完成（%s），总金额 $%.0f，持仓 %d 只",
        method, capital, len(portfolio),
    )
    return portfolio


# ---------------------------------------------------------------------------
# 再平衡计算
# ---------------------------------------------------------------------------

def calc_rebalance(
    current_holdings: pd.DataFrame,
    new_portfolio: pd.DataFrame,
    capital: float = 100_000.0,
    transaction_cost: float = 0.001,
) -> dict:
    """
    计算再平衡操作：买入 / 卖出清单与预计交易成本。

    Parameters
    ----------
    current_holdings : pd.DataFrame
        当前持仓，columns=[ticker, weight_pct]。
    new_portfolio : pd.DataFrame
        新目标持仓，columns=[ticker, weight_pct, amount_usd]。
    capital : float
        投资本金。
    transaction_cost : float
        单边交易成本比例，默认 0.1%。

    Returns
    -------
    dict
        keys: buy_list, sell_list, hold_list, turnover, estimated_cost
    """
    curr = set(current_holdings["ticker"].tolist()) if current_holdings is not None else set()
    new = set(new_portfolio["ticker"].tolist())

    buy_list = sorted(new - curr)
    sell_list = sorted(curr - new)
    hold_list = sorted(curr & new)

    # 换手率估算
    if current_holdings is not None and len(current_holdings) > 0:
        curr_w = current_holdings.set_index("ticker")["weight_pct"] / 100
        new_w = new_portfolio.set_index("ticker")["weight_pct"] / 100
        all_tickers = curr_w.index.union(new_w.index)
        curr_w = curr_w.reindex(all_tickers, fill_value=0)
        new_w = new_w.reindex(all_tickers, fill_value=0)
        turnover = (new_w - curr_w).abs().sum() / 2
    else:
        turnover = 1.0

    estimated_cost = turnover * capital * 2 * transaction_cost  # 双边

    logger.info(
        "再平衡：买入 %d 只，卖出 %d 只，保留 %d 只，换手率 %.1f%%，预计成本 $%.0f",
        len(buy_list), len(sell_list), len(hold_list),
        turnover * 100, estimated_cost,
    )

    return {
        "buy_list": buy_list,
        "sell_list": sell_list,
        "hold_list": hold_list,
        "turnover": turnover,
        "estimated_cost": estimated_cost,
    }
