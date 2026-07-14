import hashlib
import re
import unicodedata
from datetime import datetime, timedelta

import pandas as pd


NEWS_EVENT_RULES = [
    {"phrases": ["injury", "injured", "wrist", "torn"], "event_type": "injury", "performance_delta": -0.08, "availability_delta": -0.12, "uncertainty_delta": 0.12, "confidence": 0.90, "expiry_days": 28},
    {"phrases": ["illness", "sick"], "event_type": "illness", "performance_delta": -0.04, "availability_delta": -0.08, "uncertainty_delta": 0.08, "confidence": 0.80, "expiry_days": 14},
    {"phrases": ["miss", "out for", "out of", "shuts down", "suspended"], "event_type": "unavailable", "performance_delta": 0.0, "availability_delta": -0.35, "uncertainty_delta": 0.18, "confidence": 0.90, "expiry_days": 35},
    {"phrases": ["benched", "bench"], "event_type": "benching", "performance_delta": 0.0, "availability_delta": -0.70, "uncertainty_delta": 0.25, "confidence": 0.95, "expiry_days": 120},
    {"phrases": ["released", "departs", "parts ways", "leaves", "retires", "retire"], "event_type": "roster_exit", "performance_delta": 0.0, "availability_delta": -0.85, "uncertainty_delta": 0.28, "confidence": 0.95, "expiry_days": 180},
    {"phrases": ["visa"], "event_type": "visa", "performance_delta": 0.0, "availability_delta": -0.20, "uncertainty_delta": 0.15, "confidence": 0.75, "expiry_days": 35},
    {"phrases": ["returns", "back"], "event_type": "return", "performance_delta": 0.02, "availability_delta": 0.25, "uncertainty_delta": -0.10, "confidence": 0.80, "expiry_days": 28},
    {"phrases": ["signs", "joins", "adds", "promotes"], "event_type": "roster_join", "performance_delta": 0.0, "availability_delta": 0.0, "uncertainty_delta": 0.08, "confidence": 0.85, "expiry_days": 75},
    {"phrases": ["completes roster", "roster complete"], "event_type": "roster_complete", "performance_delta": 0.0, "availability_delta": 0.0, "uncertainty_delta": -0.08, "confidence": 0.75, "expiry_days": 60},
]


def normalize_name(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def contains_normalized_entity(text_norm: str, entity: str) -> bool:
    entity_norm = normalize_name(entity)
    if not entity_norm:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(entity_norm)}(?![a-z0-9])",
            text_norm,
        )
    )


def parse_news_date(value: str | float | None) -> datetime | None:
    if pd.isna(value) or not value:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.notna(parsed):
        return parsed.tz_convert(None).to_pydatetime()
    return None


def score_news_text(text: str) -> tuple[float, list[str]]:
    text_norm = normalize_name(text)
    total = 0.0
    reasons = []
    for rule in NEWS_EVENT_RULES:
        if any(contains_normalized_entity(text_norm, phrase) for phrase in rule["phrases"]):
            total += float(rule["performance_delta"]) * float(rule["confidence"])
            reasons.append(str(rule["event_type"]))
    return max(-0.20, min(0.08, total)), sorted(set(reasons))


def structure_news_events(
    news_df: pd.DataFrame,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    reference_date = reference_date or datetime.utcnow()
    events = []
    for _, article in news_df.fillna("").iterrows():
        article_text = f"{article.get('title', '')} {article.get('summary', '')}"
        normalized = normalize_name(article_text)
        published = parse_news_date(article.get("published")) or reference_date
        for rule in NEWS_EVENT_RULES:
            matched = [
                phrase
                for phrase in rule["phrases"]
                if contains_normalized_entity(normalized, phrase)
            ]
            if not matched:
                continue
            event_key = f"{article.get('url', '')}|{rule['event_type']}"
            events.append(
                {
                    "event_id": hashlib.sha256(event_key.encode("utf-8")).hexdigest()[:20],
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "article_text": article_text,
                    "published": published.isoformat(timespec="seconds"),
                    "effective_at": published.isoformat(timespec="seconds"),
                    "expires_at": (published + timedelta(days=int(rule["expiry_days"]))).isoformat(timespec="seconds"),
                    "event_type": rule["event_type"],
                    "matched_phrases": ";".join(matched),
                    "performance_delta": rule["performance_delta"],
                    "availability_delta": rule["availability_delta"],
                    "uncertainty_delta": rule["uncertainty_delta"],
                    "confidence": rule["confidence"],
                }
            )
    return pd.DataFrame(events).drop_duplicates(subset=["event_id"], keep="last") if events else pd.DataFrame()


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
    events = structure_news_events(news_df, reference_date=reference_date)
    if events.empty:
        return adjustments

    for _, event in events.fillna("").iterrows():
        article_text = event.get("article_text", "")
        article_norm = normalize_name(article_text)
        expires_at = parse_news_date(event.get("expires_at"))
        if expires_at and (reference_date or datetime.utcnow()) > expires_at:
            continue
        weight = recency_weight(event.get("published"), reference_date, recent_days)
        if weight <= 0:
            continue
        confidence = float(event.get("confidence", 0.5))

        for row in player_rows:
            team = row["team"]
            player = row["player"]
            team_hit = contains_normalized_entity(article_norm, team)
            player_hit = contains_normalized_entity(article_norm, player)
            if not team_hit and not player_hit:
                continue

            strength = 1.0 if player_hit else 0.18
            score = float(event.get("performance_delta", 0.0)) * confidence * weight * strength
            availability = float(event.get("availability_delta", 0.0)) * confidence * weight * strength
            uncertainty = float(event.get("uncertainty_delta", 0.0)) * confidence * weight * strength
            key = (team, player)
            current = adjustments.setdefault(
                key,
                {
                    "adjustment": 0.0,
                    "availability_adjustment": 0.0,
                    "uncertainty_adjustment": 0.0,
                    "reasons": [],
                },
            )
            current["adjustment"] += score
            current["availability_adjustment"] += availability
            current["uncertainty_adjustment"] += uncertainty
            current["reasons"].append(
                {
                    "title": event.get("title", ""),
                    "url": event.get("url", ""),
                    "labels": [event.get("event_type", "news")],
                    "score": round(score, 4),
                    "availability": round(availability, 4),
                    "uncertainty": round(uncertainty, 4),
                    "confidence": confidence,
                    "expires_at": event.get("expires_at", ""),
                }
            )

    for value in adjustments.values():
        value["adjustment"] = max(-0.20, min(0.08, value["adjustment"]))
        value["availability_adjustment"] = max(-0.90, min(0.30, value["availability_adjustment"]))
        value["uncertainty_adjustment"] = max(-0.15, min(0.35, value["uncertainty_adjustment"]))
    return adjustments
