"""
app.py
Streamlit 交互式 Web 界面

启动：
    streamlit run app.py

功能：
  - 用户参数输入（股票池、风险偏好、N、再平衡频率、资金量）
  - 实时因子计算与排名
  - 交互式图表（Plotly）
  - 回测结果展示
  - 结果一键下载（CSV / Excel）
"""

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.investor_profile import (
    RISK_PROFILE_LABELS,
    RISK_QUESTIONS,
    assess_investor_risk_profile,
    load_investor_profile,
    save_investor_profile,
)

logging.basicConfig(level=logging.WARNING)

CHART_BG = "#ffffff"
CHART_TEXT = "#0f172a"
CHART_GRID = "#dbe4ee"
CHART_BORDER = "#cbd5e1"


def apply_chart_theme(fig: go.Figure, *, height: int | None = None, title: str | None = None, showlegend: bool | None = None) -> go.Figure:
    """Apply a readable light theme to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_TEXT),
        legend=dict(font=dict(color=CHART_TEXT)),
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
    )
    if title is not None:
        fig.update_layout(title=dict(text=title, font=dict(color=CHART_TEXT)))
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)

    fig.update_xaxes(
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_BORDER,
        linecolor=CHART_BORDER,
        tickfont=dict(color=CHART_TEXT),
        title_font=dict(color=CHART_TEXT),
    )
    fig.update_yaxes(
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_BORDER,
        linecolor=CHART_BORDER,
        tickfont=dict(color=CHART_TEXT),
        title_font=dict(color=CHART_TEXT),
    )
    return fig

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Smart Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background: #0d1117; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        color: #58a6ff;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .metric-label { color: #f0f8ff; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    .neutral  { color: #58a6ff; }

    .section-header {
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 24px 0 12px 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6edf3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

profile_data = load_investor_profile()
if profile_data:
    profile_stamp = profile_data.get("assessed_at")
    if st.session_state.get("risk_profile_profile_stamp") != profile_stamp:
        st.session_state["risk_profile"] = profile_data["recommended_profile"]
        st.session_state["risk_profile_profile_stamp"] = profile_stamp
    st.session_state["investor_profile"] = profile_data
elif "risk_profile" not in st.session_state:
    st.session_state["risk_profile"] = "balanced"

with st.sidebar:
    st.markdown("## ⚙️ 参数配置")
    st.divider()

    universe_choice = st.selectbox(
        "股票池",
        ["S&P 500", "自定义代码列表"],
    )
    if universe_choice == "自定义代码列表":
        custom_tickers_input = st.text_area(
            "输入股票代码（每行一个，或逗号分隔）",
            value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, JNJ",
            height=120,
        )
        raw_tickers = [
            t.strip().upper()
            for t in custom_tickers_input.replace("\n", ",").split(",")
            if t.strip()
        ]
    else:
        raw_tickers = None  # 将在 run 时获取

    st.divider()
    top_n = st.slider("持仓数量 N", min_value=5, max_value=100, value=40, step=1)
    risk_profile_label = "风险偏好"
    if profile_data:
        risk_profile_label = f"风险偏好（测验推荐：{RISK_PROFILE_LABELS[profile_data['recommended_profile']]}）"
    else:
        risk_profile_label = "风险偏好（首次使用请先完成测验）"
    risk_profile = st.select_slider(
        risk_profile_label,
        options=["conservative", "balanced", "aggressive"],
        key="risk_profile",
        format_func=lambda x: {"conservative": "保守", "balanced": "平衡", "aggressive": "积极"}[x],
    )
    if profile_data:
        st.caption(f"已读取上次测验结果：{RISK_PROFILE_LABELS[profile_data['recommended_profile']]}")
    else:
        st.caption("完成主页面测验后，系统会自动保存并填充该默认值。")
    rebalance_freq = st.selectbox(
        "再平衡频率",
        ["monthly", "quarterly", "semi-annual"],
        format_func=lambda x: {"monthly": "月度", "quarterly": "季度", "semi-annual": "半年度"}[x],
    )
    weight_method = st.radio(
        "权重方案",
        ["equal_weight", "risk_parity"],
        format_func=lambda x: {"equal_weight": "等权重", "risk_parity": "风险平价"}[x],
    )
    capital = st.number_input("投资本金（美元）", min_value=1000, max_value=10_000_000,
                               value=100_000, step=10_000)
    st.divider()
    force_refresh = st.checkbox("强制刷新数据缓存", value=False)
    run_backtest_flag = st.checkbox("运行回测评估（较慢）", value=False)

    run_btn = st.button(
        "🚀 运行筛选",
        type="primary",
        use_container_width=True,
        disabled=profile_data is None,
    )

# ── Main Panel ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="padding: 20px 0 8px 0;">
      <span style="font-size:2rem; font-weight:700; color:#e6edf3;">📈 Stock Smart Screener</span>
      <span style="margin-left:12px; color:#8b949e; font-size:0.9rem;">MFT105 · 量化股票筛选系统</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if profile_data:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">投资者风险画像</div>
            <div class="metric-value">{profile_data['recommended_label']}</div>
            <div style="color:#f0f8ff; margin-top:6px;">
                得分 {profile_data['total_score']} / {profile_data['max_score']} · {profile_data['recommended_description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看测验明细", expanded=False):
        profile_df = pd.DataFrame(profile_data.get("answer_details", []))
        if not profile_df.empty:
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
        else:
            st.write("暂无测验明细。")
else:
    st.markdown("### 首次启动：投资者风险倾向测验")
    st.caption("完成后将把结果写入 data/investor_risk_profile.json，并自动作为默认风险偏好。")

    with st.form("investor_risk_survey"):
        survey_answers: dict[str, str] = {}
        for question in RISK_QUESTIONS:
            option_values = [option["value"] for option in question["options"]]
            option_labels = {option["value"]: option["label"] for option in question["options"]}
            survey_answers[question["key"]] = st.radio(
                question["label"],
                options=option_values,
                format_func=lambda value, labels=option_labels: labels[value],
                key=f"survey_{question['key']}",
            )
            st.caption(question.get("help", ""))

        submitted = st.form_submit_button("生成风险画像", type="primary", use_container_width=True)

    if submitted:
        profile_data = assess_investor_risk_profile(survey_answers)
        save_investor_profile(profile_data)
        st.session_state["investor_profile"] = profile_data
        st.rerun()

    st.info("请先完成上方测验，系统会先保存风险画像，再进入筛选页面。")
    st.stop()

if not run_btn:
    st.info("👈 在左侧配置参数后，点击「运行筛选」开始分析。")
    st.markdown(
        """
        ### 系统功能概览
        | 模块 | 说明 |
        |------|------|
        | 📊 因子计算 | 相对动量(主信号) · 动量 · 低波动 · 低估值 · 品质 |
        | 🏆 评分排名 | Z-score 标准化 + 加权综合 Factor Score |
        | 💼 组合构建 | 等权重 / 风险平价 · 持仓金额建议 |
        | 🔄 再平衡 | 月度/季度/半年度 · 换手率与交易成本估算 |
        | 📈 回测评估 | CAGR · Sharpe · 最大回撤 · 信息比率 · 月胜率 |
        """
    )
    st.stop()

# ── Pipeline 执行 ─────────────────────────────────────────────────────────────

from src.data_loader import (
    get_sp500_tickers,
    download_prices,
    download_benchmark_prices,
    download_fundamentals,
    get_risk_free_rate,
)
from src.screener import screen_stocks
from src.portfolio import build_portfolio, calc_rebalance

status = st.empty()

try:
    # 1. 获取股票池
    with st.spinner("获取股票池数据..."):
        if universe_choice == "S&P 500 (前50只，演示用)":
            tickers = get_sp500_tickers()[:50]  # 演示模式限制 50 只，加速运行
        elif universe_choice == "S&P 500":
            tickers = get_sp500_tickers()
        if raw_tickers:
            tickers = raw_tickers
        st.sidebar.caption(f"股票池: {len(tickers)} 只")

    # 2. 下载数据
    with st.spinner("下载价格与财务数据..."):
        prices = download_prices(
            tickers=tickers,
            start="2015-01-01",
            end="2024-12-31",
            force_refresh=force_refresh,
        )
        benchmark_prices = download_benchmark_prices(
            ticker="^GSPC",
            start="2015-01-01",
            end="2024-12-31",
            force_refresh=force_refresh,
        )
        fundamentals = download_fundamentals(tickers=tickers, force_refresh=force_refresh)
        rf_rate = get_risk_free_rate()

    # IS 价格
    prices_is = prices.loc["2015-01-01":"2022-12-31"]

    # 3. 因子计算
    with st.spinner("计算因子分数..."):
        top_n_df, all_scores = screen_stocks(
            prices=prices_is,
            fundamentals=fundamentals,
            benchmark_prices=benchmark_prices.loc["2015-01-01":"2022-12-31"],
            top_n=top_n,
            risk_profile=risk_profile,
        )

    # 4. 组合构建
    with st.spinner("构建持仓组合..."):
        portfolio = build_portfolio(
            top_n_df=top_n_df,
            prices=prices_is,
            method=weight_method,
            capital=capital,
        )

    status.success("✅ 分析完成！")

except Exception as e:
    st.error(f"❌ 运行失败: {e}")
    st.exception(e)
    st.stop()

# ── 展示结果 ──────────────────────────────────────────────────────────────────

# ---------- 概览指标卡 ----------
st.markdown('<div class="section-header">📊 筛选结果概览</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_score = top_n_df["factor_score"].max()
    st.metric("最高 Factor Score", f"{top_score:.3f}")

with col2:
    st.metric("持仓股票数", f"{len(top_n_df)} 只")

with col3:
    if "sector" in top_n_df.columns:
        sectors = top_n_df["sector"].nunique()
        st.metric("覆盖行业数", f"{sectors} 个")
    else:
        st.metric("投资本金", f"${capital:,.0f}")

with col4:
    avg_score = top_n_df["factor_score"].mean()
    st.metric("平均 Factor Score", f"{avg_score:.3f}")

st.divider()

# ---------- Tab 布局 ----------
tab1, tab2, tab3, tab4 = st.tabs(["🏆 因子排名", "💼 持仓组合", "📈 回测评估", "📥 数据下载"])

# ─── Tab 1: 因子排名 ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Top 20 Factor Score 排名**")
        plot_df = all_scores.head(20).copy()
        if "ticker" not in plot_df.columns:
            plot_df = plot_df.reset_index()
            plot_df.columns = ["ticker" if c == "index" else c for c in plot_df.columns]

        colors = ["#f85149" if i < top_n else "#58a6ff" for i in range(len(plot_df))]
        fig_bar = go.Figure(go.Bar(
            x=plot_df["factor_score"],
            y=plot_df["ticker"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in plot_df["factor_score"]],
            textposition="outside",
            textfont=dict(color=CHART_TEXT),
        ))
        fig_bar = apply_chart_theme(fig_bar, height=max(400, len(plot_df) * 28))
        fig_bar.update_layout(xaxis_title="Factor Score", margin={"t": 10, "b": 40})
        fig_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("**Top N 明细**")
        display_cols = ["ticker", "factor_score", "relative_momentum_z", "momentum_z", "volatility_z", "value_z", "quality_z"]
        if "sector" in top_n_df.columns:
            display_cols.append("sector")
        st.dataframe(
            top_n_df[display_cols].style.format({
                "factor_score": "{:.3f}",
                "relative_momentum_z": "{:.2f}",
                "momentum_z": "{:.2f}",
                "volatility_z": "{:.2f}",
                "value_z": "{:.2f}",
                "quality_z": "{:.2f}",
            }).set_properties(**{"background-color": CHART_BG, "color": CHART_TEXT}),
            use_container_width=True,
            height=500,
        )

    # 雷达图
    st.markdown("**四维因子雷达图（Top N）**")
    factor_specs = [
        ("relative_momentum_z", "相对动量 Relative Momentum"),
        ("momentum_z", "动量 Momentum"),
        ("volatility_z", "低波动 Volatility"),
        ("value_z", "低估值 Value"),
        ("quality_z", "品质 Quality"),
    ]
    factor_z_cols = [c for c, _ in factor_specs if c in top_n_df.columns]
    factor_labels = [label for c, label in factor_specs if c in top_n_df.columns]

    radar_df = top_n_df.set_index("ticker")[factor_z_cols] if "ticker" in top_n_df.columns else top_n_df[factor_z_cols]
    radar_df = radar_df.fillna(0).head(10)

    fig_radar = go.Figure()
    for ticker, row in radar_df.iterrows():
        vals = row.tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=factor_labels + [factor_labels[0]],
            fill="toself",
            name=ticker,
            opacity=0.7,
        ))
    fig_radar = apply_chart_theme(fig_radar, height=500)
    fig_radar.update_layout(
        polar={
            "bgcolor": CHART_BG,
            "radialaxis": {"visible": True, "tickfont": {"color": CHART_TEXT}, "gridcolor": CHART_GRID},
            "angularaxis": {"tickfont": {"color": CHART_TEXT}},
        },
        margin={"t": 20},
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 因子分布箱线图
    st.markdown("**四大因子 Z-score 分布**")
    scores_reset = all_scores.reset_index() if "ticker" not in all_scores.columns else all_scores
    fig_box = go.Figure()
    colors_box = ["#f85149", "#58a6ff", "#3fb950", "#ffa657", "#8b949e"]
    for col, label, color in zip(factor_z_cols, factor_labels, colors_box):
        if col in all_scores.columns:
            fig_box.add_trace(go.Box(
                y=all_scores[col].dropna(),
                name=label,
                marker_color=color,
                boxmean="sd",
            ))
    fig_box.update_layout(
        height=400,
        showlegend=False,
        yaxis_title="Z-score",
    )
    fig_box = apply_chart_theme(fig_box, height=400)
    st.plotly_chart(fig_box, use_container_width=True)


# ─── Tab 2: 持仓组合 ──────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(f"**持仓权重（{{'equal_weight':'等权重','risk_parity':'风险平价'}}[weight_method]）**".replace(
            "{{'equal_weight':'等权重','risk_parity':'风险平价'}}[weight_method]",
            {"equal_weight": "等权重", "risk_parity": "风险平价"}[weight_method]
        ))
        fig_pie = px.pie(
            portfolio,
            values="weight_pct",
            names="ticker",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_pie.update_traces(textfont=dict(color=CHART_TEXT))
        fig_pie = apply_chart_theme(fig_pie, height=420)
        fig_pie.update_layout(margin={"t": 20}, legend={"font": {"size": 11, "color": CHART_TEXT}})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("**持仓明细与建议金额**")
        fmt_cols = {"weight_pct": "{:.2f}%", "amount_usd": "${:,.0f}",
                    "latest_price": "${:.2f}", "shares_approx": "{:.1f}"}
        st.dataframe(
            portfolio.style.format(fmt_cols).set_properties(**{"background-color": CHART_BG, "color": CHART_TEXT}),
            use_container_width=True,
            height=420,
        )

    # 再平衡提示（演示：假设上期持仓为空）
    st.markdown("**再平衡建议（首次建仓）**")
    empty_holdings = pd.DataFrame(columns=["ticker", "weight_pct"])
    rebalance_info = calc_rebalance(
        current_holdings=empty_holdings,
        new_portfolio=portfolio,
        capital=capital,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("需买入", f"{len(rebalance_info['buy_list'])} 只")
    c2.metric("需卖出", f"{len(rebalance_info['sell_list'])} 只")
    c3.metric("预计交易成本", f"${rebalance_info['estimated_cost']:,.0f}")

    if rebalance_info["buy_list"]:
        st.success("买入：" + "  ·  ".join(rebalance_info["buy_list"]))


# ─── Tab 3: 回测评估 ──────────────────────────────────────────────────────────
with tab3:
    if not run_backtest_flag:
        st.info("💡 在左侧勾选「运行回测评估」后重新点击运行，以查看回测结果。")
    else:
        from src.backtester import run_backtest

        with st.spinner("运行 Out-of-Sample 回测（2023-2024）..."):
            try:
                results_oos = run_backtest(
                    prices=prices,
                    fundamentals=fundamentals,
                    benchmark_prices=benchmark_prices,
                    start="2023-01-01",
                    end="2024-12-31",
                    top_n=top_n,
                    risk_profile=risk_profile,
                    rebalance_freq=rebalance_freq,
                    weight_method=weight_method,
                    risk_free_rate=rf_rate,
                )
                m = results_oos["metrics"]

                # 指标卡
                st.markdown("**Out-of-Sample 绩效指标（2023–2024）**")
                cols = st.columns(5)
                cols[0].metric("策略 CAGR", f"{m['strategy_cagr']:.1%}",
                                delta=f"{(m['strategy_cagr']-m['baseline_cagr']):.1%} vs Baseline")
                cols[1].metric("Sharpe Ratio", f"{m['strategy_sharpe']:.2f}",
                                delta=f"{(m['strategy_sharpe']-m['baseline_sharpe']):.2f} vs Baseline")
                cols[2].metric("最大回撤", f"{m['strategy_max_drawdown']:.1%}")
                cols[3].metric("信息比率 IR", f"{m['information_ratio']:.2f}")
                cols[4].metric("月度胜率", f"{m['monthly_win_rate']:.1%}")

                # 净值曲线
                nav_df = pd.DataFrame({
                    "策略组合": results_oos["strategy_nav"],
                    "S&P 500 Baseline": results_oos["baseline_nav"],
                })
                fig_nav = px.line(
                    nav_df, x=nav_df.index, y=["策略组合", "S&P 500 Baseline"],
                    color_discrete_map={"策略组合": "#f85149", "S&P 500 Baseline": "#58a6ff"},
                    title="OOS 净值曲线（2023–2024）",
                )
                fig_nav = apply_chart_theme(fig_nav, height=420)
                fig_nav.update_layout(xaxis_title="日期", yaxis_title="净值")
                st.plotly_chart(fig_nav, use_container_width=True)

                # Baseline 对比表
                st.markdown("**策略 vs. Baseline 对比**")
                compare_df = pd.DataFrame({
                    "指标": ["CAGR", "Sharpe Ratio", "最大回撤", "年化波动率"],
                    "策略": [f"{m['strategy_cagr']:.2%}", f"{m['strategy_sharpe']:.2f}",
                             f"{m['strategy_max_drawdown']:.2%}", f"{m['strategy_annualized_vol']:.2%}"],
                    "Baseline": [f"{m['baseline_cagr']:.2%}", f"{m['baseline_sharpe']:.2f}",
                                 f"{m['baseline_max_drawdown']:.2%}", f"{m['baseline_annualized_vol']:.2%}"],
                })
                st.dataframe(
                    compare_df.style.set_properties(**{"background-color": CHART_BG, "color": CHART_TEXT}),
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:
                st.error(f"回测失败: {e}")
                st.exception(e)


# ─── Tab 4: 数据下载 ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("**下载分析结果**")

    # Top N CSV
    csv_top_n = top_n_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 下载 Top N 股票（CSV）",
        data=csv_top_n,
        file_name="top_n_stocks.csv",
        mime="text/csv",
    )

    # 全量因子分 CSV
    csv_all = all_scores.reset_index().to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 下载全量因子分（CSV）",
        data=csv_all,
        file_name="all_factor_scores.csv",
        mime="text/csv",
    )

    # 持仓权重 CSV
    csv_port = portfolio.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 下载持仓权重（CSV）",
        data=csv_port,
        file_name="portfolio_weights.csv",
        mime="text/csv",
    )

    # Excel 汇总
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        top_n_df.to_excel(writer, sheet_name="Top N Stocks", index=False)
        all_scores.reset_index().to_excel(writer, sheet_name="All Factor Scores", index=False)
        portfolio.to_excel(writer, sheet_name="Portfolio Weights", index=False)
    st.download_button(
        "📥 下载汇总 Excel",
        data=buf.getvalue(),
        file_name="stock_screener_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()
    st.caption("⚠️ 本系统为课程作业，所有输出均不构成任何实际投资建议。")
