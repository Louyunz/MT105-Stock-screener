# 📈 Stock Smart Screener

> MFT105 · Python 及其在金融中的应用 | 量化股票筛选系统

---

## 项目简介

本系统以 Python 为核心技术栈，整合多源公开金融数据，提供**基于量化规则的股票筛选、评分排名与可操作决策建议**，降低个人投资者的选股门槛。

当前版本还集成了**AI 投顾（DeepSeek）**：在本地输入 API Key 后，系统会基于样本外回测结果自动生成面向普通投资者的文字建议。API 只在当前会话中使用，不会写入文件或缓存。

---

## 快速开始

### 0. Python 版本

本项目建议 Python 版本为 **3.12.0**，以避免跨设备部署时出现依赖不兼容。

- 云端/平台部署文件：`runtime.txt`

建议先确认解释器版本：

```bash
python --version
```

输出应为 `Python 3.12.0`（或至少同 minor 版本 `3.12.x`）。
若无对应版本python，可以使用安装脚本一键安装python环境。

### 1. Windows 一键部署（推荐）

在 Windows 上可直接双击 `deploy.bat`，或在终端执行：

```bash
deploy.bat
```

该脚本会自动完成：
- 检查/安装 Python 3.12.x
- 创建并激活 `.venv`
- 安装 `requirements.txt`
- 启动 Streamlit 应用

仅部署不启动应用：

```bash
deploy.bat --no-run
```



### 2. 手动部署（跨平台）

如果你不使用 Windows 一键脚本，可按以下步骤手动部署：

```bash
# 建议先创建并激活虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```


### 3. 一键运行完整 Pipeline（CLI）

请先完成上面的任一种部署流程（手动部署或 `deploy.bat`）。

若你是手动部署，请先激活 `.venv` 后再执行以下命令。

```bash
# 基础用法（S&P 500，Top 10，月度再平衡）
python run_pipeline.py --universe sp500 --top_n 10 --rebalance monthly

# 保守风格 + 风险平价权重
python run_pipeline.py --universe sp500 --top_n 15 \
    --risk_profile conservative --weight_method risk_parity --capital 500000

# 仅运行 Out-of-Sample 评估（不含 IS 回测，更快）
python run_pipeline.py --universe sp500 --oos_only

# 自定义股票池
python run_pipeline.py --universe "AAPL,MSFT,GOOGL,NVDA,META,TSLA,JPM,JNJ,V,PG" --top_n 5

# 全参数说明
python run_pipeline.py --help
```

### 4. 启动交互式 Web 界面

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，在侧边栏配置参数后点击「运行筛选」。
首次打开时会先完成一次风险倾向测验，结果会保存到 `data/investor_risk_profile.json`，后续启动会自动读取并作为风险偏好默认值。

如果你希望启用 AI 投顾，请在侧边栏的「AI 投顾配置」中填写 DeepSeek API Key，并补充提示词要求。填写后，回测评估页右侧会自动启用 AI 投顾，基于当前回测结果生成建议；API 不会被保存到本地。

AI 投顾建议主要面向普通投资者，输出通常包含：是否值得跟随策略、风险提示、仓位/再平衡建议和下一步操作建议。
**AI可能会出错，AI输出的建议仅供参考**

---

## 目录结构

```
stock-screener/
├── data/                   # 缓存数据与输出结果
│   ├── prices.parquet      # 价格缓存
│   ├── fundamentals.parquet
│   ├── top_n_stocks.csv    # 筛选结果
│   ├── all_factor_scores.csv
│   ├── portfolio_weights.csv
│   ├── oos_nav_curve.csv
│   └── charts/             # 图表输出
├── notebooks/              # 探索性分析（EDA）
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # 数据获取与缓存
│   ├── factors.py          # 四大因子计算
│   ├── screener.py         # 评分、排名与筛选主逻辑
│   ├── portfolio.py        # 组合构建与权重计算
│   ├── backtester.py       # 回测引擎
│   └── visualizer.py       # 图表生成
├── app.py                  # Streamlit 主程序
├── run_pipeline.py         # 命令行一键运行入口
├── deploy.bat              # Windows 一键部署脚本
├── .python-version         # 本地 Python 版本锁定（3.12.0）
├── runtime.txt             # 部署平台 Python 版本锁定（python-3.12.0）
├── requirements.txt
└── README.md
```

---

## 因子设计

如需更完整的策略说明，请先阅读 [strategy.md](strategy.md)。

| 因子维度 | 指标 | 计算方式 | 默认权重 | 方向 |
|---------|------|---------|---------|------|
| **相对动量 Relative Momentum** | 12-1 月超额动量 | 个股 12-1 月动量减去 S&P 500 12-1 月动量 | 主信号 | 越高越好 ↑ |
| **动量 Momentum** | 12-1 月动量 | 过去 12 个月收益率（剔除最近 1 月） | 辅助 | 越高越好 ↑ |
| **波动 Volatility** | 年化波动率 | 过去 252 日日收益率标准差 × √252 | 辅助 | 越低越好 ↓ |
| **估值 Value** | P/E、P/B 综合 | 行业相对 P/E 与 P/B 排名取均值 | 辅助 | 越低越好 ↓ |
| **品质 Quality** | ROE | ROE 截尾后标准化 | 辅助 | 越高越好 ↑ |

所有因子均经过：Winsorize（1%–99%）→ Z-score 标准化 → 方向对齐。

> 说明：当回测或界面提供 S&P 500 指数基准数据时，系统会优先使用“相对动量”作为主选股信号；其余因子作为辅助过滤。

---

## 风险偏好权重预设

> 当提供 S&P 500 基准数据时，系统默认使用“相对动量”作为主信号；下表仅在未提供基准数据的回退模式中生效。

| 风险偏好 | 动量 | 波动 | 估值 | 品质 |
|---------|------|------|------|------|
| 保守 Conservative | 20% | 35% | 25% | 20% |
| 平衡 Balanced | 30% | 25% | 25% | 20% |
| 积极 Aggressive | 45% | 15% | 20% | 20% |

---

## 评估框架

| 阶段 | 期间 | 用途 |
|------|------|------|
| In-Sample | 2015–2022 | 因子参数确定、权重设计 |
| Out-of-Sample | 2023–2024 | 独立评估，不得回溯修改规则 |

**主要评估指标：** CAGR、Sharpe Ratio、最大回撤、信息比率（IR）、月度胜率

---

## 数据来源

- **价格数据**：Yahoo Finance via `yfinance`（复权收盘价）
- **财务数据**：Yahoo Finance via `yfinance.Ticker.info`（P/E、P/B、ROE）
- **无风险利率**：FRED 3M T-Bill via `pandas_datareader`
- **成分股列表**：Wikipedia（S&P 500）

---

## 已知限制

- 财务数据通常滞后 1–3 个月，实盘存在执行时差
- 使用当前成分股名单，存在一定幸存者偏差
- 交易成本已按固定比例计入回测（默认 0.1%/单边），未计入市场冲击成本
- **本系统为课程作业，所有输出均不构成任何实际投资建议**
- AI 投顾依赖 DeepSeek 在线接口，若 API Key 无效、网络受限或接口不可用，将无法生成建议；系统会提示具体错误信息
- API Key 仅用于当前会话，不会写入项目文件、缓存或日志

---

## 可重现性

```bash
# 固定随机种子已在 run_pipeline.py 中设置（RANDOM_SEED = 42）
# 方式 A（Windows）：
deploy.bat --no-run

# 方式 B（手动，跨平台）：
# 1) 创建并激活 .venv
# 2) 安装依赖
pip install -r requirements.txt

# 运行主流程
python run_pipeline.py --universe sp500 --top_n 10 --rebalance monthly
```

### AI 投顾使用说明

1. 启动 `streamlit run app.py`
2. 在左侧侧边栏填写 DeepSeek API Key 和提示词要求
3. 点击「运行筛选」并在需要时勾选「运行回测评估」
4. 回测结果完成后，右侧 AI 投顾会自动给出建议

建议在提示词中明确你的关注点，例如：回撤控制、稳健性、是否适合普通投资者、以及是否只做小仓位试运行。

---

*MFT105 · 2026 Spring*
