"""
investor_profile.py
投资者风险倾向测验与持久化。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = ROOT / "data" / "investor_risk_profile.json"

RISK_PROFILE_LABELS = {
    "conservative": "保守",
    "balanced": "平衡",
    "aggressive": "积极",
}

RISK_PROFILE_DESCRIPTIONS = {
    "conservative": "优先控制回撤与本金安全，适合更稳健的配置。",
    "balanced": "在波动与收益之间保持均衡，适合大多数投资者。",
    "aggressive": "愿意承受更高波动以追求更高长期收益。",
}

RISK_QUESTIONS = [
    {
        "key": "investment_horizon",
        "label": "你的投资期限更接近哪种？",
        "help": "投资期限越长，通常可以承受更高波动。",
        "options": [
            {"value": "under_1y", "label": "1 年以内", "score": 0},
            {"value": "one_to_three_y", "label": "1-3 年", "score": 1},
            {"value": "three_to_five_y", "label": "3-5 年", "score": 2},
            {"value": "over_5y", "label": "5 年以上", "score": 3},
        ],
    },
    {
        "key": "drawdown_tolerance",
        "label": "你能接受的单次回撤大约是多少？",
        "help": "这里的回撤是指组合短期从高点到低点的下跌幅度。",
        "options": [
            {"value": "under_5pct", "label": "5% 以内", "score": 0},
            {"value": "around_10pct", "label": "10% 左右", "score": 1},
            {"value": "around_20pct", "label": "20% 左右", "score": 2},
            {"value": "over_30pct", "label": "30% 以上", "score": 3},
        ],
    },
    {
        "key": "reaction_to_drop",
        "label": "如果市场短期下跌 20%，你更可能怎么做？",
        "help": "这用于衡量你在压力场景下的真实行为。",
        "options": [
            {"value": "sell_all", "label": "尽快离场", "score": 0},
            {"value": "reduce_position", "label": "减仓观望", "score": 1},
            {"value": "hold", "label": "继续持有", "score": 2},
            {"value": "buy_more", "label": "分批加仓", "score": 3},
        ],
    },
    {
        "key": "investment_goal",
        "label": "你的主要投资目标更接近哪种？",
        "help": "目标越偏增长，通常可以接受更高波动。",
        "options": [
            {"value": "capital_preservation", "label": "保本与现金流", "score": 0},
            {"value": "steady_growth", "label": "稳步增值", "score": 1},
            {"value": "balanced_growth", "label": "稳健增长", "score": 2},
            {"value": "high_growth", "label": "长期高增长", "score": 3},
        ],
    },
    {
        "key": "experience",
        "label": "你的投资经验更接近哪种？",
        "help": "经验越多，通常对波动的理解也越成熟。",
        "options": [
            {"value": "beginner", "label": "新手", "score": 0},
            {"value": "under_3y", "label": "1-3 年", "score": 1},
            {"value": "three_to_five_y", "label": "3-5 年", "score": 2},
            {"value": "over_5y", "label": "5 年以上", "score": 3},
        ],
    },
]


def load_investor_profile() -> dict | None:
    """读取已保存的风险测验结果。"""
    if not PROFILE_PATH.exists():
        return None

    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(profile, dict):
        return None
    if "recommended_profile" not in profile:
        return None
    return profile


def save_investor_profile(profile: dict) -> None:
    """将风险测验结果写入本地文件。"""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)


def assess_investor_risk_profile(answers: dict[str, str]) -> dict:
    """根据问卷答案生成风险画像。"""
    answer_details: list[dict[str, object]] = []
    total_score = 0
    max_score = 0

    for question in RISK_QUESTIONS:
        options_by_value = {option["value"]: option for option in question["options"]}
        selected_value = answers.get(question["key"], question["options"][0]["value"])
        selected_option = options_by_value.get(selected_value, question["options"][0])

        option_max_score = max(int(option["score"]) for option in question["options"])
        total_score += int(selected_option["score"])
        max_score += option_max_score
        answer_details.append(
            {
                "question": question["label"],
                "answer": selected_option["label"],
                "score": int(selected_option["score"]),
            }
        )

    score_ratio = total_score / max_score if max_score else 0.0
    if score_ratio <= 0.33:
        recommended_profile = "conservative"
    elif score_ratio <= 0.67:
        recommended_profile = "balanced"
    else:
        recommended_profile = "aggressive"

    return {
        "version": 1,
        "assessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_score": total_score,
        "max_score": max_score,
        "score_pct": round(score_ratio * 100, 1),
        "recommended_profile": recommended_profile,
        "recommended_label": RISK_PROFILE_LABELS[recommended_profile],
        "recommended_description": RISK_PROFILE_DESCRIPTIONS[recommended_profile],
        "answer_details": answer_details,
        "answers": answers,
    }