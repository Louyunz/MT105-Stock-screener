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
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

WEIGHT_METHOD_LABELS = {
    "equal_weight": "等权重",
    "risk_parity": "风险平价",
}


def _add_metric(col, label: str, value: str, help_text: str, delta: str | None = None):
    if delta is None:
        col.metric(label, value, help=help_text)
    else:
        col.metric(label, value, delta=delta, help=help_text)


def _build_ai_advice_prompt(
    results_oos: dict,
    top_n_df: pd.DataFrame,
    portfolio: pd.DataFrame,
    risk_profile: str,
    weight_method: str,
    capital: float,
    backtest_start: str,
    backtest_end: str,
) -> str:
    metrics = results_oos["metrics"]
    top_stocks = top_n_df[[c for c in ["ticker", "factor_score", "sector"] if c in top_n_df.columns]].head(8)
    portfolio_view = portfolio[[c for c in ["ticker", "weight_pct", "amount_usd"] if c in portfolio.columns]].head(8)

    return (
        "你是一名面向普通投资者的量化投顾，只能基于给定的回测与组合数据给出保守、可执行的建议。"
        "不要承诺收益，不要夸大，不要给出个股买卖点，只能给出仓位、风控、再平衡和观察要点。\n\n"
        f"投资者风险偏好: {RISK_PROFILE_LABELS.get(risk_profile, risk_profile)}\n"
        f"组合方式: {WEIGHT_METHOD_LABELS.get(weight_method, weight_method)}\n"
        f"本金假设: {capital:,.0f} 美元\n"
        f"样本外时间: {backtest_start} 到 {backtest_end}\n"
        f"策略CAGR: {metrics['strategy_cagr']:.2%}\n"
        f"基准CAGR: {metrics['baseline_cagr']:.2%}\n"
        f"策略Sharpe: {metrics['strategy_sharpe']:.2f}\n"
        f"基准Sharpe: {metrics['baseline_sharpe']:.2f}\n"
        f"策略最大回撤: {metrics['strategy_max_drawdown']:.2%}\n"
        f"基准最大回撤: {metrics['baseline_max_drawdown']:.2%}\n"
        f"信息比率: {metrics['information_ratio']:.2f}\n"
        f"月度胜率: {metrics['monthly_win_rate']:.2%}\n\n"
        f"Top N 股票摘要:\n{top_stocks.to_string(index=False)}\n\n"
        f"组合权重摘要:\n{portfolio_view.to_string(index=False)}\n\n"
        "请输出以下四部分：1) 一句话结论；2) 对普通投资者是否值得跟随；3) 风险提示；4) 下一步操作建议。"
    )


def _call_deepseek(api_key: str, prompt: str) -> str:
    api_key = (api_key or "").strip().strip('"').strip("'")
    if api_key.lower().startswith("bearer "):
        # 用户可能把 "Bearer <key>" 整段贴进输入框，这里自动兼容。
        api_key = api_key.split(" ", 1)[1].strip()

    if not api_key:
        raise RuntimeError("DeepSeek API Key 为空，请在侧边栏输入有效的 sk- 开头密钥。")

    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "DeepSeek API Key 包含非 ASCII 字符。请仅粘贴原始密钥（通常以 sk- 开头），"
            "不要包含中文说明、全角符号或额外文本。"
        ) from exc

    url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个谨慎、透明、面向普通投资者的量化投顾。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"DeepSeek 接口返回错误: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"DeepSeek 网络请求失败: {exc.reason}") from exc

    choices = response_data.get("choices", [])
    if not choices:
        raise RuntimeError("DeepSeek 未返回可用建议。")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("DeepSeek 返回内容为空。")
    return content


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

    .insight-box {
        background: linear-gradient(135deg, rgba(88,166,255,0.12), rgba(31,111,235,0.06));

        border-radius: 12px;
        padding: 16px 18px;
        margin: 12px 0 16px 0;
        color: #3d7bc2;
    }
    .insight-box ul {
        margin: 10px 0 0 18px;
        padding: 0;
    }
    .insight-box li {
        margin: 6px 0;
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
    custom_date_range = st.checkbox("自定义回测日期范围", value=False)
    if custom_date_range:
        default_end_date = pd.to_datetime(time.strftime("%Y-%m-%d"))
        default_start_date = default_end_date - pd.DateOffset(years=5)
        start_date = st.date_input("回测开始日期", value=default_start_date)
        end_date = st.date_input("回测结束日期", value=pd.to_datetime(time.strftime("%Y-%m-%d")))
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            st.stop()
    st.divider()
    top_n = st.slider("持仓数量 N", min_value=1, max_value=20, value=5, step=1)
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

    st.divider()
    st.markdown("### 🤖 AI 投顾配置")
    st.caption("这些输入只保存在当前会话中，不会写入文件或缓存。建议在点击运行前先填写。")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="sk-...",
        help="仅用于本次会话的 AI 投顾调用，不会保存到本地。",
        key="deepseek_api_key_input",
    )
    ai_prompt_requirement = st.text_area(
        "提示词要求",
        value=(
            "请用普通投资者能看懂的方式回答，只给出可执行建议；"
            "不要承诺收益，不要夸大结果，不要输出个股买卖点；"
            "请围绕仓位、风险、再平衡和是否值得跟随策略进行分析。"
        ),
        height=140,
        help="你可以补充希望 AI 特别关注的角度，例如回撤、稳定性或仓位控制。",
        key="ai_prompt_requirement",
    )

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
        | 🤖 AI投顾 | 基于回测结果的智能投资建议 |
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
        if custom_date_range:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
        else:
            end_str = time.strftime("%Y-%m-%d")
            start_str = (pd.to_datetime(end_str) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
        prices = download_prices(
            tickers=tickers,
            start=start_str,
            end=end_str,
            force_refresh=force_refresh,
        )
        benchmark_prices = download_benchmark_prices(
            ticker="^GSPC",
            start=start_str,
            end=end_str,
            force_refresh=force_refresh,
        )
        fundamentals = download_fundamentals(tickers=tickers, force_refresh=force_refresh)
        rf_rate = get_risk_free_rate()

    # IS 价格
    prices_is = prices.loc[start_str:end_str]

    # 3. 因子计算
    with st.spinner("计算因子分数..."):
        top_n_df, all_scores = screen_stocks(
            prices=prices_is,
            fundamentals=fundamentals,
            benchmark_prices=benchmark_prices.loc[start_str:end_str],
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
st.caption("鼠标悬停在下方指标上可以查看解释；这些指标不是单纯展示结果，而是帮助普通投资者判断是否值得继续看回测和组合建议。")

col1, col2, col3, col4 = st.columns(4)

with col1:
    top_score = top_n_df["factor_score"].max()
    _add_metric(
        col1,
        "最高 Factor Score",
        f"{top_score:.3f}",
        "综合得分越高，说明股票在相对动量、波动率、估值与品质上的综合表现越好。",
    )

with col2:
    _add_metric(
        col2,
        "持仓股票数",
        f"{len(top_n_df)} 只",
        "最终纳入建议组合的股票数量。数量越少，组合越集中；数量越多，通常越分散。",
    )

with col3:
    if "sector" in top_n_df.columns:
        sectors = top_n_df["sector"].nunique()
        _add_metric(
            col3,
            "覆盖行业数",
            f"{sectors} 个",
            "建议组合覆盖的行业数量。覆盖越广，单一行业波动对组合影响通常越小。",
        )
    else:
        _add_metric(
            col3,
            "投资本金",
            f"${capital:,.0f}",
            "用于测算持仓金额和再平衡结果的假设本金。",
        )

with col4:
    avg_score = top_n_df["factor_score"].mean()
    _add_metric(
        col4,
        "平均 Factor Score",
        f"{avg_score:.3f}",
        "Top N 股票的平均综合得分。这个值越高，说明整组股票整体越强。",
    )

top_three = "、".join(top_n_df["ticker"].head(3).tolist()) if "ticker" in top_n_df.columns else "无"
st.markdown(
    f"""
    <div class="insight-box">
        <strong>给投资者的建议</strong>
        <ul>
            <li>当前风格：{RISK_PROFILE_LABELS.get(risk_profile, risk_profile)}，组合方式：{WEIGHT_METHOD_LABELS.get(weight_method, weight_method)}。</li>
            <li>优先关注的前 3 只股票：{top_three}。</li>
            <li>如果你更在意稳健，建议优先看低波动或风险平价版本；如果你更在意上涨弹性，可以优先看综合得分最高的标的。</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---------- Tab 布局 ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 因子排名", "💼 持仓组合", "📈 回测评估", "🤖AI投顾","📥 数据下载"])

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
            hovertemplate="股票：%{y}<br>综合得分：%{x:.3f}<extra>Factor Score</extra>",
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
    results_oos = None
    left_col, right_col = st.columns([3, 2])

    with left_col:
        if not run_backtest_flag:
            st.info("💡 在左侧勾选「运行回测评估」后重新点击运行，以查看回测结果。")
            st.caption("DeepSeek API Key 和提示词要求请先在侧边栏填写；只有在回测运行后才能基于结果给出建议。")
        else:
            from src.backtester import run_backtest

            with st.spinner(f"运行 Out-of-Sample 回测（{start_str} 到 {end_str}）..."):
                try:
                    results_oos = run_backtest(
                        prices=prices,
                        fundamentals=fundamentals,
                        benchmark_prices=benchmark_prices,
                        start=start_str,
                        end=end_str,
                        top_n=top_n,
                        risk_profile=risk_profile,
                        rebalance_freq=rebalance_freq,
                        weight_method=weight_method,
                        risk_free_rate=rf_rate,
                    )
                    m = results_oos["metrics"]

                    # 指标卡
                    st.markdown(f"**Out-of-Sample 绩效指标（{start_str} 到 {end_str}）**")
                    st.info(
                        f"验证机制说明：这里使用样本外 OOS（{start_str} 到 {end_str}）进行检验，并与 S&P 500 基准对照。"
                        "重点不是单看收益高低，而是同时看 CAGR、Sharpe、最大回撤、信息比率和月度胜率。"
                        "如果策略在 OOS 中不能稳定优于基准，就只能把它当作筛选参考，而不是直接重仓依据。"
                    )
                    cols = st.columns(5)
                    cols[0].metric(
                        "策略 CAGR",
                        f"{m['strategy_cagr']:.1%}",
                        delta=f"{(m['strategy_cagr']-m['baseline_cagr']):.1%} vs Baseline",
                        help="年化复合收益率，回答“长期下来能赚多少”。",
                    )
                    cols[1].metric(
                        "Sharpe Ratio",
                        f"{m['strategy_sharpe']:.2f}",
                        delta=f"{(m['strategy_sharpe']-m['baseline_sharpe']):.2f} vs Baseline",
                        help="单位波动带来的超额收益，回答“赚得是否值得承担波动”。",
                    )
                    cols[2].metric(
                        "最大回撤",
                        f"{m['strategy_max_drawdown']:.1%}",
                        help="历史上最差从高点跌到低点的幅度，回答“最坏会跌多少”。",
                    )
                    cols[3].metric(
                        "信息比率 IR",
                        f"{m['information_ratio']:.2f}",
                        help="策略相对基准的超额收益稳定性，越高说明越常跑赢基准。",
                    )
                    cols[4].metric(
                        "月度胜率",
                        f"{m['monthly_win_rate']:.1%}",
                        help="按月比较策略与基准谁更好，反映相对表现的稳定程度。",
                    )

                    # 净值曲线
                    nav_df = pd.DataFrame({
                        "策略组合": results_oos["strategy_nav"],
                        "S&P 500 Baseline": results_oos["baseline_nav"],
                    })
                    fig_nav = px.line(
                        nav_df, x=nav_df.index, y=["策略组合", "S&P 500 Baseline"],
                        color_discrete_map={"策略组合": "#f85149", "S&P 500 Baseline": "#58a6ff"},
                        title=f"OOS 净值曲线（{start_str} – {end_str}）",
                    )
                    fig_nav = apply_chart_theme(fig_nav, height=420)
                    fig_nav.update_layout(xaxis_title="日期", yaxis_title="净值")
                    fig_nav.update_traces(
                        hovertemplate="日期：%{x|%Y-%m-%d}<br>净值：%{y:.3f}<extra>%{fullData.name}</extra>",
                    )
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

                    st.markdown("**对普通投资者的解读**")
                    if m["strategy_cagr"] >= m["baseline_cagr"] and m["strategy_sharpe"] >= m["baseline_sharpe"]:
                        st.success("策略在样本外同时具备更高收益和更好的风险调整后表现，可作为进一步小规模验证的候选方案。")
                    elif m["strategy_max_drawdown"] > m["baseline_max_drawdown"]:
                        st.warning("策略收益未必差，但回撤更大，普通投资者应优先控制仓位，避免直接满仓使用。")
                    else:
                        st.info("策略更适合作为筛选参考，而不是直接重仓方案；建议结合风险偏好和资金规模做二次筛选。")

                    st.markdown(
                        """
                        <div class="insight-box">
                            <strong>验证结果的含义</strong>
                            <ul>
                                <li>CAGR：看长期复合收益，回答“能不能赚到钱”。</li>
                                <li>Sharpe：看单位波动带来的收益，回答“值不值得承担波动”。</li>
                                <li>最大回撤：看最坏情况下可能跌多少，回答“我能不能扛住”。</li>
                                <li>信息比率和月度胜率：看策略相对基准是否稳定胜出。</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"回测失败: {e}")
                    st.exception(e)

    with tab4:
        st.markdown("### 🤖 AI 投顾")
        st.caption("在侧边栏填写 DeepSeek API Key 后会自动启用；API 仅在当前会话内使用，不会写入本地。")

        if not api_key:
            st.info("请先在侧边栏填写 DeepSeek API Key 和提示词要求。")
        elif not run_backtest_flag:
            st.info("已检测到 API Key。请先运行回测，AI 投顾会在结果可用后自动生成建议。")
        elif results_oos is None:
            st.warning("当前没有可用的回测结果，请先确保回测成功完成。")
        else:
            try:
                prompt = _build_ai_advice_prompt(
                    results_oos,
                    top_n_df,
                    portfolio,
                    risk_profile,
                    weight_method,
                    capital,
                    start_str,
                    end_str,
                )
                if ai_prompt_requirement.strip():
                    prompt = ai_prompt_requirement.strip() + "\n\n" + prompt

                cache = st.session_state.setdefault("ai_advisor_cache", {})
                api_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
                cache_key = hashlib.sha256((api_fingerprint + "::" + prompt).encode("utf-8")).hexdigest()

                if cache_key not in cache:
                    with st.spinner("正在调用 DeepSeek 生成建议..."):
                        cache[cache_key] = _call_deepseek(api_key, prompt)

                ai_reply = cache[cache_key]
                if isinstance(ai_reply, str) and ai_reply.startswith("AI 投顾调用失败："):
                    st.error(ai_reply)
                else:
                    st.markdown("**AI 投顾建议**")
                    st.write(ai_reply)

                st.caption("AI 投顾的建议基于回测结果和当前组合构建方案，结合了你在提示词中强调的关注点。请务必理解这些建议是基于历史数据和模型推断的，并不保证未来表现；投资决策仍需谨慎。")
            except Exception as e:
                st.error(f"AI 投顾调用失败：{e}")


# ─── Tab 5: 数据下载 ──────────────────────────────────────────────────────────
with tab5:
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
