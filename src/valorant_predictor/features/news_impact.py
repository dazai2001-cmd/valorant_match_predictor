import re
from datetime import datetime

import pandas as pd


NEWS_RULES = [
    ("injury", -0.14, "injury"),
    ("injured", -0.14, "injury"),
    ("wrist", -0.12, "injury"),
    ("torn", -0.12, "injury"),
    ("illness", -0.10, "illness"),
    ("sick", -0.10, "illness"),
    ("miss", -0.09, "availability"),
    ("out for", -0.09, "availability"),
    ("out of", -0.07, "availability"),
    ("shuts down", -0.14, "availability"),
    ("benched", -0.09, "benching"),
    ("bench", -0.08, "benching"),
    ("released", -0.08, "roster loss"),
    ("departs", -0.08, "roster loss"),
    ("parts ways", -0.08, "roster loss"),
    ("leaves", -0.07, "roster loss"),
    ("retires", -0.12, "retirement"),
    ("retire", -0.12, "retirement"),
    ("visa", -0.06, "availability"),
    ("suspended", -0.12, "availability"),
    ("returns", 0.07, "return"),
    ("back", 0.04, "return"),
    ("signs", 0.05, "signing"),
    ("joins", 0.05, "signing"),
    ("adds", 0.04, "signing"),
    ("promotes", 0.04, "promotion"),
    ("completes", 0.04, "roster stability"),
    ("qualify", 0.03, "momentum"),
    ("secures", 0.03, "momentum"),
    ("wins", 0.03, "momentum"),
]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def parse_news_date(value: str | float | None) -> datetime | None:
    if pd.isna(value) or not value:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def score_news_text(text: str) -> tuple[float, list[str]]:
    text_norm = normalize_name(text)
    total = 0.0
    reasons = []
    for phrase, score, label in NEWS_RULES:
        if normalize_name(phrase) in text_norm:
            total += score
            reasons.append(label)
    return max(-0.25, min(0.16, total)), sorted(set(reasons))


def recency_weight(published: str | None, reference_date: datetime | None, recent_days: int) -> float:
    if not reference_date:
        reference_date = datetime.utcnow()
    published_date = parse_news_date(published)
    if not published_date:
        return 0.65

    age_days = max(0, (reference_date.date() - published_date.date()).days)
    if age_days > recent_days:
        return 0.0
    return 1.0 - (age_days / recent_days * 0.55)


def player_news_adjustments(
    news_df: pd.DataFrame,
    players: pd.DataFrame,
    recent_days: int = 45,
    reference_date: datetime | None = None,
) -> dict[tuple[str, str], dict]:
    adjustments: dict[tuple[str, str], dict] = {}
    if news_df.empty or players.empty:
        return adjustments

    player_rows = players[["team", "player"]].drop_duplicates().to_dict("records")

    for _, article in news_df.fillna("").iterrows():
        article_text = f"{article.get('title', '')} {article.get('summary', '')}"
        article_norm = normalize_name(article_text)
        base_score, labels = score_news_text(article_text)
        if base_score == 0:
            continue

        weight = recency_weight(article.get("published"), reference_date, recent_days)
        if weight <= 0:
            continue

        for row in player_rows:
            team = row["team"]
            player = row["player"]
            team_hit = normalize_name(team) in article_norm
            player_hit = normalize_name(player) in article_norm
            if not team_hit and not player_hit:
                continue

            strength = 1.0 if player_hit else 0.22
            score = base_score * weight * strength
            key = (team, player)
            current = adjustments.setdefault(key, {"adjustment": 0.0, "reasons": []})
            current["adjustment"] += score
            current["reasons"].append(
                {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "labels": labels,
                    "score": round(score, 4),
                }
            )

    for value in adjustments.values():
        value["adjustment"] = max(-0.30, min(0.18, value["adjustment"]))
    return adjustments
