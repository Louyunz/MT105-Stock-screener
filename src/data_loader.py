"""
data_loader.py
数据获取与缓存模块
- 从 yfinance 拉取日频价格数据
- 从 yfinance 拉取财务比率数据
- 从 FRED 拉取无风险利率
- 本地 CSV 缓存，避免重复下载
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import bs4 
from io import StringIO

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# S&P 500 成分股获取
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """
    从 Wikipedia 获取 S&P 500 当前成分股列表。

    Returns
    -------
    list[str]
        股票代码列表，已排序去重。
    """

    cache_path = os.path.join(CACHE_DIR, "sp500_tickers.csv")
    if os.path.exists(cache_path):
        stocks = pd.read_csv(cache_path, header=None)[0].tolist()
        logger.info("从缓存加载 S&P 500 成分股 (%d 只)", len(stocks))
        return stocks

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
        resp = requests.get(url,headers=headers)
        resp.text.encode('utf-8')
        #tables = pd.read_html(resp.text)
        # tables = pd.read_html(url)
        #df = tables[0]
        soup = bs4.BeautifulSoup(StringIO(resp.text), "html.parser")
        table = soup.find('table', {'id': 'constituents'})
        stocks = []
        if table:
            # 遍历表格的每一行
            for row in table.find('tbody').find_all('tr')[1:]: # 跳过表头
                cols = row.find_all('td')
                if len(cols) >= 2:
                    symbol = cols[0].text.strip() # 股票代码
                    company = cols[1].text.strip() # 公司名称
                    stocks.append(symbol)

        #tickers = df["Symbol"].str.replace(".", "-", regex=False).sort_values().tolist()
        pd.Series(stocks).to_csv(cache_path, index=False, header=False)
        logger.info("已获取 S&P 500 成分股 (%d 只)", len(stocks))
        return stocks
    except Exception as e:
        logger.error("获取 S&P 500 成分股失败: %s", e)
        raise


# ---------------------------------------------------------------------------
# 价格数据
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    下载复权日频收盘价，返回宽表（行=日期，列=股票代码）。

    Parameters
    ----------
    tickers : list[str]
        股票代码列表。
    start : str
        起始日期 YYYY-MM-DD。
    end : str
        截止日期 YYYY-MM-DD。
    force_refresh : bool
        强制重新下载，忽略本地缓存。

    Returns
    -------
    pd.DataFrame
        复权收盘价宽表。
    """
    cache_path = os.path.join(CACHE_DIR, "prices.parquet")
    if os.path.exists(cache_path) and not force_refresh:
        prices = pd.read_parquet(cache_path)
        logger.info("从缓存加载价格数据，shape=%s", prices.shape)
        return prices

    logger.info("开始下载价格数据，共 %d 只股票...", len(tickers))
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=True,
            threads=True,
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices.index = pd.to_datetime(prices.index)
        prices.to_parquet(cache_path)
        logger.info("价格数据下载完成，shape=%s", prices.shape)
        return prices
    except Exception as e:
        logger.error("价格数据下载失败: %s", e)
        raise


def download_benchmark_prices(
    ticker: str = "^GSPC",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.Series:
    """
    下载基准指数复权收盘价（默认 S&P 500 指数）。

    Parameters
    ----------
    ticker : str
        基准指数代码，默认 '^GSPC'。
    start : str
        起始日期 YYYY-MM-DD。
    end : str
        截止日期 YYYY-MM-DD。
    force_refresh : bool
        强制重新下载，忽略本地缓存。

    Returns
    -------
    pd.Series
        基准指数复权收盘价序列，index=日期。
    """
    safe_ticker = ticker.replace("^", "").replace("=", "_")
    cache_path = os.path.join(CACHE_DIR, f"benchmark_{safe_ticker}.parquet")
    if os.path.exists(cache_path) and not force_refresh:
        benchmark = pd.read_parquet(cache_path).squeeze()
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark.name = ticker
        logger.info("从缓存加载基准价格数据 %s，长度=%d", ticker, len(benchmark))
        return benchmark

    logger.info("开始下载基准指数价格数据: %s", ticker)
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        benchmark = raw["Close"] if "Close" in raw.columns else raw.squeeze()
        if isinstance(benchmark, pd.DataFrame):
            # yfinance 在部分版本下会返回单列 DataFrame，这里统一压平成 Series
            if benchmark.shape[1] == 0:
                raise ValueError(f"基准指数 {ticker} 未返回有效收盘价列。")
            benchmark = benchmark.iloc[:, 0]
        benchmark = benchmark.dropna()
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark.name = ticker

        benchmark.to_frame(name="close").to_parquet(cache_path)
        logger.info("基准价格数据下载完成 %s，长度=%d", ticker, len(benchmark))
        return benchmark
    except Exception as e:
        logger.error("基准价格数据下载失败 (%s): %s", ticker, e)
        raise


# ---------------------------------------------------------------------------
# 财务数据
# ---------------------------------------------------------------------------

def download_fundamentals(
    tickers: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    批量获取财务比率（P/E、P/B、ROE、EPS 历史）。

    Returns
    -------
    pd.DataFrame
        index=ticker，columns=[pe_ratio, pb_ratio, roe, eps_list]
    """
    cache_path = os.path.join(CACHE_DIR, "fundamentals.parquet")
    if os.path.exists(cache_path) and not force_refresh:
        fund = pd.read_parquet(cache_path)
        logger.info("从缓存加载财务数据，shape=%s", fund.shape)
        return fund

    logger.info("开始获取财务数据，共 %d 只股票...", len(tickers))
    records = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            logger.info("  进度: %d / %d", i + 1, len(tickers))
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", np.nan),
            })
        except Exception as e:
            logger.warning("  [%s] 财务数据获取失败: %s", ticker, e)
            records.append({
                "ticker": ticker,
                "pe_ratio": np.nan,
                "pb_ratio": np.nan,
                "roe": np.nan,
                "sector": "Unknown",
                "market_cap": np.nan,
            })

    fund = pd.DataFrame(records).set_index("ticker")
    fund.to_parquet(cache_path)
    logger.info("财务数据获取完成，shape=%s", fund.shape)
    return fund


# ---------------------------------------------------------------------------
# 无风险利率
# ---------------------------------------------------------------------------

def get_risk_free_rate() -> float:
    """
    获取年化无风险利率（3M T-Bill）。
    优先尝试从 FRED 获取，失败则返回默认值 0.05。

    Returns
    -------
    float
        年化无风险利率（小数，如 0.05 代表 5%）。
    """
    # try:
    import pandas_datareader.data as web
    from datetime import datetime, timedelta

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=30)
    tb = web.DataReader("DTB3", "fred", start_dt, end_dt)
    rate = tb.dropna().iloc[-1, 0] / 100.0
    logger.info("无风险利率（3M T-Bill）= %.4f", rate)
    return float(rate)
    # except Exception as e:
    #     logger.warning("无法从 FRED 获取无风险利率 (%s)，使用默认值 0.05", e)
    #     return 0.05



if __name__ == "__main__":
    # tickers = get_sp500_tickers()
    # print(tickers[:10])
    # prices = download_prices(tickers[:10], force_refresh=True)
    # print(prices)
    # fundamentals = download_fundamentals(tickers[:10], force_refresh=True)
    # print(fundamentals)
    risk_free_rate = get_risk_free_rate()
    print(f"Risk-free rate: {risk_free_rate:.4f}")