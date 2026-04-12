"""
run_pipeline.py
命令行一键运行入口

用法示例：
    python run_pipeline.py --universe sp500 --top_n 10 --rebalance monthly
    python run_pipeline.py --universe sp500 --top_n 15 --risk_profile conservative \
                           --weight_method risk_parity --capital 500000
    python run_pipeline.py --universe sp500 --oos_only   # 仅跑 OOS 评估
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path


import numpy as np
import pandas as pd

# 固定随机种子，确保可重现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    get_sp500_tickers,
    download_prices,
    download_benchmark_prices,
    download_fundamentals,
    get_risk_free_rate,
)
from src.screener import screen_stocks
from src.portfolio import build_portfolio
from src.backtester import run_backtest
from src.visualizer import (
    plot_factor_scores,
    plot_portfolio_weights,
    plot_nav_curve,
    plot_factor_distribution,
    plot_radar,
)

# 样本期间配置
IN_SAMPLE_START = "2015-01-01"
IN_SAMPLE_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2024-12-31"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stock Smart Screener — 量化股票筛选系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--universe", default="sp500",
                        help="股票池：'sp500' 或逗号分隔的自定义代码列表")
    parser.add_argument("--top_n", type=int, default=40, help="持仓数量 N")
    parser.add_argument("--risk_profile", default="balanced",
                        choices=["conservative", "balanced", "aggressive"],
                        help="风险偏好")
    parser.add_argument("--rebalance", default="monthly",
                        choices=["monthly", "quarterly", "semi-annual"],
                        help="再平衡频率")
    parser.add_argument("--weight_method", default="equal_weight",
                        choices=["equal_weight", "risk_parity"],
                        help="权重分配方案")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="投资本金（美元）")
    parser.add_argument("--force_refresh", action="store_true",
                        help="强制重新下载数据，忽略缓存")
    parser.add_argument("--oos_only", action="store_true",
                        help="仅运行 Out-of-Sample 回测评估")
    parser.add_argument("--no_charts", action="store_true",
                        help="跳过图表生成（加速运行）")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Stock Smart Screener Pipeline 启动")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: 获取股票池
    # ------------------------------------------------------------------
    logger.info("[Step 1] 获取股票池...")
    if args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = [t.strip().upper() for t in args.universe.split(",")]
    logger.info("股票池大小: %d 只", len(tickers))

    # ------------------------------------------------------------------
    # Step 2: 下载数据
    # ------------------------------------------------------------------
    logger.info("[Step 2] 下载价格与财务数据...")
    prices = download_prices(
        tickers=tickers,
        start=IN_SAMPLE_START,
        end=OOS_END,
        force_refresh=args.force_refresh,
    )
    benchmark_prices = download_benchmark_prices(
        ticker="^GSPC",
        start=IN_SAMPLE_START,
        end=OOS_END,
        force_refresh=args.force_refresh,
    )
    fundamentals = download_fundamentals(
        tickers=tickers,
        force_refresh=args.force_refresh,
    )
    rf_rate = get_risk_free_rate()

    # ------------------------------------------------------------------
    # Step 3: 因子计算 & 筛选（基于 IS 数据）
    # ------------------------------------------------------------------
    logger.info("[Step 3] 因子计算与股票筛选（In-Sample 数据）...")
    prices_is = prices.loc[IN_SAMPLE_START:IN_SAMPLE_END]
    benchmark_is = benchmark_prices.loc[IN_SAMPLE_START:IN_SAMPLE_END]

    top_n_df, all_scores = screen_stocks(
        prices=prices_is,
        fundamentals=fundamentals,
        benchmark_prices=benchmark_is,
        top_n=args.top_n,
        risk_profile=args.risk_profile,
    )

    logger.info("Top %d 股票:", args.top_n)
    pd.set_option("display.float_format", "{:.3f}".format)
    display_cols = ["ticker", "factor_score", "relative_momentum_z", "momentum_z", "volatility_z",
                    "value_z", "quality_z"]
    display_cols = [c for c in display_cols if c in top_n_df.columns]
    print(top_n_df[display_cols].to_string(index=False))

    # 导出 CSV
    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    top_n_df.to_csv(output_dir / "top_n_stocks.csv", index=False)
    all_scores.reset_index().to_csv(output_dir / "all_factor_scores.csv", index=False)
    logger.info("因子分结果已导出至 data/")

    # ------------------------------------------------------------------
    # Step 4: 组合构建
    # ------------------------------------------------------------------
    logger.info("[Step 4] 构建持仓组合（%s）...", args.weight_method)
    portfolio = build_portfolio(
        top_n_df=top_n_df,
        prices=prices_is,
        method=args.weight_method,
        capital=args.capital,
    )
    print("\n持仓权重建议：")
    print(portfolio.to_string(index=False))
    portfolio.to_csv(output_dir / "portfolio_weights.csv", index=False)

    # ------------------------------------------------------------------
    # Step 5: 回测评估
    # ------------------------------------------------------------------
    logger.info("[Step 5] 运行回测评估...")

    results_is = None
    if not args.oos_only:
        logger.info("  In-Sample 回测：%s ~ %s", IN_SAMPLE_START, IN_SAMPLE_END)
        try:
            results_is = run_backtest(
                prices=prices,
                fundamentals=fundamentals,
                benchmark_prices=benchmark_prices,
                start=IN_SAMPLE_START,
                end=IN_SAMPLE_END,
                top_n=args.top_n,
                risk_profile=args.risk_profile,
                rebalance_freq=args.rebalance,
                weight_method=args.weight_method,
                risk_free_rate=rf_rate,
            )
            _print_metrics("In-Sample", results_is["metrics"])
        except Exception as e:
            logger.error("  IS 回测失败: %s", e)

    logger.info("  Out-of-Sample 回测：%s ~ %s", OOS_START, OOS_END)
    try:
        results_oos = run_backtest(
            prices=prices,
            fundamentals=fundamentals,
            benchmark_prices=benchmark_prices,
            start=OOS_START,
            end=OOS_END,
            top_n=args.top_n,
            risk_profile=args.risk_profile,
            rebalance_freq=args.rebalance,
            weight_method=args.weight_method,
            risk_free_rate=rf_rate,
        )
        _print_metrics("Out-of-Sample", results_oos["metrics"])

        # 导出 OOS 净值曲线
        nav_df = pd.DataFrame({
            "strategy_nav": results_oos["strategy_nav"],
            "baseline_nav": results_oos["baseline_nav"],
        })
        nav_df.to_csv(output_dir / "oos_nav_curve.csv")

        # 导出 OOS 指标
        pd.DataFrame([results_oos["metrics"]]).to_csv(
            output_dir / "oos_metrics.csv", index=False
        )

    except Exception as e:
        logger.error("  OOS 回测失败: %s", e)
        results_oos = None

    # ------------------------------------------------------------------
    # Step 6: 可视化
    # ------------------------------------------------------------------
    if not args.no_charts:
        logger.info("[Step 6] 生成图表...")
        try:
            plot_factor_scores(all_scores, top_n=20)
            plot_portfolio_weights(portfolio)
            plot_factor_distribution(all_scores)
            plot_radar(top_n_df)
            if results_oos:
                plot_nav_curve(
                    results_oos["strategy_nav"],
                    results_oos["baseline_nav"],
                    results_oos["metrics"],
                )
            logger.info("图表已保存至 data/charts/")
        except Exception as e:
            logger.warning("图表生成出错: %s", e)

    logger.info("=" * 60)
    logger.info("Pipeline 完成！所有结果已保存至 data/ 目录。")
    logger.info("=" * 60)


def _print_metrics(label: str, metrics: dict):
    print(f"\n{'=' * 40}")
    print(f"  {label} 绩效指标")
    print(f"{'=' * 40}")
    print(f"  策略 CAGR         : {metrics['strategy_cagr']:.2%}")
    print(f"  Baseline CAGR     : {metrics['baseline_cagr']:.2%}")
    print(f"  策略 Sharpe       : {metrics['strategy_sharpe']:.2f}")
    print(f"  Baseline Sharpe   : {metrics['baseline_sharpe']:.2f}")
    print(f"  策略 最大回撤      : {metrics['strategy_max_drawdown']:.2%}")
    print(f"  Baseline 最大回撤  : {metrics['baseline_max_drawdown']:.2%}")
    print(f"  信息比率 IR        : {metrics['information_ratio']:.2f}")
    print(f"  月度胜率           : {metrics['monthly_win_rate']:.1%}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
