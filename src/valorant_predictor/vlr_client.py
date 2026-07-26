import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data_quality import (
    classify_event_tier,
    enrich_match_metadata,
    estimated_patch_epoch,
    infer_event_region,
)
from .config import (
    BASE_URL,
    EXCLUDED_MATCHES_CSV,
    MATCH_COVERAGE_CSV,
    MATCHES_CSV,
    NEWS_CSV,
    NEWS_EVENTS_CSV,
    REQUEST_HEADERS,
    ROSTERS_CSV,
    TEAM_PAGES,
    UPCOMING_MATCHES_CSV,
    VLRGGAPI_BASE_URL,
)
from .features.form_calculations import filter_matches_by_time, normalize_map_name
from .vlrggapi_client import (
    enrich_match_records_with_map_metadata,
    fetch_vlrggapi_match_metadata,
    map_metadata_from_veto,
    vlrggapi_is_healthy,
)


MATCH_COLUMNS = [
    "match_url",
    "match_id",
    "match_date",
    "season_year",
    "event_id",
    "event_name",
    "event_series",
    "event_stage",
    "event_tier",
    "event_region",
    "is_lan",
    "patch",
    "patch_source",
    "map_veto",
    "match_importance",
    "competition_tier",
    "competition_strength_weight",
    "team_tier_at_match",
    "opponent_tier_at_match",
    "team_promoted_next_season",
    "opponent_promoted_next_season",
    "data_source",
    "map_id",
    "map_number",
    "map_name",
    "map_pick_team",
    "map_pick_type",
    "map_veto_order",
    "map_data_source",
    "stat_scope",
    "team",
    "team_id",
    "team_region",
    "opponent",
    "opponent_id",
    "opponent_region",
    "winner",
    "winner_id",
    "player",
    "player_id",
    "agents",
    "vlr_rating",
    "acs",
    "kills",
    "deaths",
    "assists",
    "is_winner",
    "team_score",
    "opp_score",
    "map_team_score",
    "map_opp_score",
    "scraped_at",
]

COVERAGE_COLUMNS = [
    "team",
    "has_url",
    "candidate_urls",
    "matches",
    "matches_total",
    "legacy_matches",
    "season_year",
    "rows",
    "target_matches",
    "missing_matches",
    "status",
]

UPCOMING_COLUMNS = [
    "match_id",
    "match_url",
    "match_date",
    "team1",
    "team2",
    "event_name",
    "event_series",
    "event_stage",
    "event_tier",
    "is_lan",
    "best_of",
    "scraped_at",
]

TEAM_NAME_ALIASES = {
    "pcificesports": "PCFIC Esports",
    "jdgesports": "JD Gaming",
    "jdmalljdgesportsjdgesports": "JD Gaming",
    "kiwoomdrx": "DRX",
    "wuxititanesportsclubtitanesportsclub": "Titan Esports Club",
}


def infer_event_tier(event_name: str) -> str:
    return classify_event_tier(event_name)


def infer_match_importance(event_name: str, event_stage: str) -> float:
    event_text = str(event_name or "").lower()
    stage_text = str(event_stage or "").lower()
    importance = 1.0
    if any(token in event_text for token in ["valorant champions", "masters", "esports world cup"]):
        importance += 0.05
    if "grand final" in stage_text:
        importance += 0.10
    elif "final" in stage_text:
        importance += 0.07
    elif any(token in stage_text for token in ["playoff", "quarterfinal", "semifinal", "decider"]):
        importance += 0.04
    return min(1.15, importance)


def infer_is_lan(event_name: str) -> bool:
    text = str(event_name or "").lower()
    return any(token in text for token in ["valorant champions", "masters", "esports world cup"])


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def absolute_url(href: str) -> str:
    return urljoin(BASE_URL, href)


def canonical_match_url(url: str) -> str:
    if url is None or pd.isna(url) or not str(url).strip():
        return ""
    parts = urlsplit(absolute_url(str(url).strip()))
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def match_id_from_url(url: str) -> int | None:
    match = re.search(r"vlr\.gg/(\d+)|/(\d+)/", str(url))
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def team_key(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.lower())


def canonical_team_lookup(team_pages: dict[str, str]) -> dict[str, str]:
    return {team_key(team): team for team in team_pages if team_key(team)}


def canonical_team_id_lookup(team_pages: dict[str, str]) -> dict[str, str]:
    lookup = {}
    for team, url in team_pages.items():
        match = re.search(r"/team/matches/(\d+)/", str(url))
        if match:
            lookup[match.group(1)] = team
    return lookup


def _id_text(value) -> str:
    if value is None or pd.isna(value) or value == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def canonical_team_name(value: str | None, lookup: dict[str, str]) -> str | None:
    if value is None or pd.isna(value):
        return value

    text = str(value).strip()
    key = team_key(text)
    if key in lookup:
        return lookup[key]

    alias = TEAM_NAME_ALIASES.get(key)
    if alias and alias in lookup.values():
        return alias

    for known_key, known_name in lookup.items():
        if len(known_key) < 4 or len(key) < 4:
            continue
        if known_key in key or key in known_key:
            return known_name
    return text


def canonicalize_match_records(records: list[dict], team_pages: dict[str, str]) -> list[dict]:
    lookup = canonical_team_lookup(team_pages)
    id_lookup = canonical_team_id_lookup(team_pages)
    if not lookup:
        return records

    for record in records:
        for field, id_field in [
            ("team", "team_id"),
            ("opponent", "opponent_id"),
            ("winner", "winner_id"),
        ]:
            team_id = record.get(id_field)
            canonical = id_lookup.get(_id_text(team_id))
            record[field] = canonical or canonical_team_name(record.get(field), lookup)
        if record.get("map_pick_team"):
            record["map_pick_team"] = canonical_team_name(record["map_pick_team"], lookup)
    return records


def canonicalize_match_dataframe(matches: pd.DataFrame, team_pages: dict[str, str]) -> pd.DataFrame:
    if matches.empty:
        return matches

    lookup = canonical_team_lookup(team_pages)
    id_lookup = canonical_team_id_lookup(team_pages)
    if not lookup:
        return matches

    output = matches.copy()
    for field, id_field in [
        ("team", "team_id"),
        ("opponent", "opponent_id"),
        ("winner", "winner_id"),
    ]:
        if field in output.columns:
            canonical_names = output[field].apply(lambda value: canonical_team_name(value, lookup))
            if id_field in output.columns:
                id_names = output[id_field].apply(lambda value: id_lookup.get(_id_text(value)))
                output[field] = id_names.where(id_names.notna(), canonical_names)
            else:
                output[field] = canonical_names
    if "map_pick_team" in output.columns:
        output["map_pick_team"] = output["map_pick_team"].apply(
            lambda value: canonical_team_name(value, lookup)
            if value is not None and not pd.isna(value) and str(value).strip()
            else ""
        )
    return output


def match_date_from_page(soup: BeautifulSoup) -> str | None:
    for node in soup.select("[data-utc-ts]"):
        value = node.get("data-utc-ts")
        if value:
            return value.strip()

    for node in soup.select(".moment-tz-convert, .match-header-date"):
        value = node.get_text(" ", strip=True)
        if value:
            return value
    return None


def parse_team_match_candidates_page(
    soup: BeautifulSoup,
    limit: int | None = None,
) -> list[dict]:
    candidates = []
    seen_match_ids = set()
    links = soup.select("a.wf-card.fc-flex.m-item[href]")
    if not links:
        links = soup.select("a.wf-card.m-item[href]")

    for link in links:
        href = link.get("href", "")
        if not href.startswith("/") or "vs" not in href:
            continue
        full_url = canonical_match_url(href)
        match_id = match_id_from_url(full_url)
        if match_id is None or match_id in seen_match_ids:
            continue

        event_node = link.select_one(".m-item-event")
        event_parts = list(event_node.stripped_strings) if event_node else []
        team_names = [
            node.get_text(" ", strip=True)
            for node in link.select(".m-item-team-name")[:2]
        ]
        date_node = link.select_one(".m-item-date")
        date_text = date_node.get_text(" ", strip=True) if date_node else ""
        date_match = re.search(
            r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b",
            date_text,
        )
        match_date = (
            f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-"
            f"{int(date_match.group(3)):02d}"
            if date_match
            else ""
        )

        seen_match_ids.add(match_id)
        candidates.append(
            {
                "match_id": match_id,
                "match_url": full_url,
                "match_date": match_date,
                "event_name": event_parts[0] if event_parts else "",
                "teams": team_names,
            }
        )
        if limit is not None and limit > 0 and len(candidates) >= limit:
            break
    return candidates


def get_team_match_candidates(
    session: requests.Session,
    team_url: str,
    limit: int | None = None,
) -> list[dict]:
    return parse_team_match_candidates_page(get_soup(session, team_url), limit)


def get_team_match_urls(
    session: requests.Session,
    team_url: str,
    limit: int | None = None,
) -> list[str]:
    return [
        candidate["match_url"]
        for candidate in get_team_match_candidates(session, team_url, limit)
    ]


def filter_tier1_match_candidates(
    candidates: list[dict],
    season_year: int | None,
    tier1_teams: Iterable[str],
) -> tuple[list[dict], dict[str, int]]:
    team_lookup = canonical_team_lookup({team: "" for team in tier1_teams})
    eligible_team_keys = set(team_lookup)
    kept = []
    rejected = {
        "wrong_season": 0,
        "missing_date": 0,
        "explicit_non_tier1_event": 0,
        "non_tier1_matchup": 0,
    }

    for candidate in candidates:
        parsed_date = pd.to_datetime(candidate.get("match_date"), utc=True, errors="coerce")
        if season_year is not None:
            if pd.isna(parsed_date):
                rejected["missing_date"] += 1
                continue
            if int(parsed_date.year) != int(season_year):
                rejected["wrong_season"] += 1
                continue

        event_tier = classify_event_tier(candidate.get("event_name", ""))
        if event_tier in {"tier2", "game_changers"}:
            rejected["explicit_non_tier1_event"] += 1
            continue
        if event_tier == "promotion":
            kept.append(candidate)
            continue

        team_keys = {
            team_key(canonical_team_name(team, team_lookup))
            for team in candidate.get("teams", [])
            if team_key(team)
        }
        if len(team_keys) == 2 and team_keys.issubset(eligible_team_keys):
            kept.append(candidate)
        else:
            rejected["non_tier1_matchup"] += 1

    return kept, rejected


def normalize_match_dataframe(
    matches: pd.DataFrame,
    team_pages: dict[str, str] | None = None,
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    output = matches.copy()
    for column in MATCH_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    output["match_url"] = output["match_url"].apply(canonical_match_url)
    output["map_name"] = output["map_name"].apply(normalize_map_name)
    parsed_ids = output["match_url"].apply(match_id_from_url)
    output["match_id"] = pd.to_numeric(output["match_id"], errors="coerce").fillna(parsed_ids)
    for column in [
        "event_id",
        "season_year",
        "map_id",
        "map_number",
        "map_veto_order",
        "team_id",
        "opponent_id",
        "winner_id",
        "player_id",
        "team_score",
        "opp_score",
        "map_team_score",
        "map_opp_score",
        "match_importance",
        "competition_strength_weight",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["match_importance"] = output["match_importance"].fillna(1.0).clip(0.75, 1.15)
    output["competition_strength_weight"] = (
        output["competition_strength_weight"].fillna(1.0).clip(0.25, 1.25)
    )
    parsed_dates = pd.to_datetime(output["match_date"], utc=True, errors="coerce")
    output["season_year"] = output["season_year"].fillna(parsed_dates.dt.year)
    for column, default in {
        "competition_tier": "unknown",
        "team_tier_at_match": "unknown",
        "opponent_tier_at_match": "unknown",
        "data_source": "legacy",
    }.items():
        output[column] = output[column].fillna("").astype(str).str.strip().replace("", default)
    for column in ["team_promoted_next_season", "opponent_promoted_next_season"]:
        output[column] = output[column].fillna(False).astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    output["map_pick_type"] = (
        output["map_pick_type"].fillna("").astype(str).str.strip().str.lower().replace("", "unknown")
    )
    output["map_data_source"] = (
        output["map_data_source"].fillna("").astype(str).str.strip().replace("", "legacy")
    )
    output = enrich_match_metadata(output)

    scope = output["stat_scope"].fillna("").astype(str).str.strip().str.lower()
    output["stat_scope"] = scope.where(scope != "", output["map_id"].notna().map({True: "map", False: "legacy"}))
    if team_pages:
        output = canonicalize_match_dataframe(output, team_pages)

    output = output[output["match_id"].notna() & (output["match_url"] != "")].copy()
    return output.reindex(columns=MATCH_COLUMNS)


def partition_registry_tier1_history(
    matches: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty or registry.empty or not {"team", "opponent"}.issubset(matches.columns):
        return matches.copy(), pd.DataFrame(columns=matches.columns)

    eligible_registry = registry.copy()
    if "tier" in eligible_registry:
        eligible_registry = eligible_registry[
            eligible_registry["tier"].fillna("").astype(str).str.lower().eq("tier1")
        ]
    if "active" in eligible_registry:
        active = eligible_registry["active"].fillna(True).astype(str).str.lower()
        eligible_registry = eligible_registry[active.isin({"true", "1", "yes"})]
    eligible_registry["_year"] = pd.to_numeric(
        eligible_registry.get("season_year"),
        errors="coerce",
    )
    eligible_registry = eligible_registry[eligible_registry["_year"].notna()].copy()
    if eligible_registry.empty:
        return matches.copy(), pd.DataFrame(columns=matches.columns)

    registry_by_year = {
        int(year): {team_key(name) for name in group["team"].dropna().astype(str)}
        for year, group in eligible_registry.groupby("_year")
    }
    output = matches.copy()
    dates = pd.to_datetime(
        output.get("match_date", pd.Series(pd.NaT, index=output.index)),
        utc=True,
        errors="coerce",
    )
    years = pd.to_numeric(
        output.get("season_year", pd.Series(float("nan"), index=output.index)),
        errors="coerce",
    ).fillna(dates.dt.year)
    competition_tiers = output.get(
        "competition_tier",
        pd.Series("unknown", index=output.index),
    ).fillna("unknown").astype(str).str.lower()
    event_tiers = output.get(
        "event_tier",
        pd.Series("unknown", index=output.index),
    ).fillna("unknown").astype(str).str.lower()
    explicit_non_tier1 = competition_tiers.isin({"tier2", "game_changers"}) | event_tiers.isin(
        {"tier2", "game_changers"}
    )
    promotion = competition_tiers.eq("promotion") | event_tiers.eq("promotion")
    team_keys = output["team"].map(team_key)
    opponent_keys = output["opponent"].map(team_key)
    has_registry = pd.Series(False, index=output.index)
    tier1_matchup = pd.Series(False, index=output.index)
    for year, names in registry_by_year.items():
        year_mask = years.eq(year)
        has_registry |= year_mask
        tier1_matchup |= year_mask & team_keys.isin(names) & opponent_keys.isin(names)

    keep = ~has_registry | promotion | (tier1_matchup & ~explicit_non_tier1)
    inferred_tier1 = has_registry & tier1_matchup & ~explicit_non_tier1 & ~promotion
    unknown_competition = competition_tiers.isin({"", "unknown", "nan"})
    output.loc[inferred_tier1 & unknown_competition, "competition_tier"] = "tier1"
    output.loc[inferred_tier1, "team_tier_at_match"] = "tier1"
    output.loc[inferred_tier1, "opponent_tier_at_match"] = "tier1"
    output.loc[inferred_tier1 & unknown_competition, "competition_strength_weight"] = 1.0

    relevant = output[keep].copy().reset_index(drop=True)
    excluded = output[~keep].copy().reset_index(drop=True)
    relevant.attrs.update(matches.attrs)
    relevant.attrs["excluded_non_tier1_rows"] = len(excluded)
    relevant.attrs["excluded_non_tier1_matches"] = _unique_match_count(excluded)
    return relevant, excluded


def dedupe_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    output = matches.copy()
    scoped_keys = []
    for _, row in output.iterrows():
        match_id = _id_text(row.get("match_id"))
        map_id = _id_text(row.get("map_id"))
        team_identity = _id_text(row.get("team_id")) or team_key(row.get("team", ""))
        player = _id_text(row.get("player_id")) or str(row.get("player", "")).strip().lower()
        if map_id:
            scoped_keys.append(f"map:{match_id}:{map_id}:{team_identity}:{player}")
            continue

        signature = ":".join(
            str(row.get(column, ""))
            for column in ["agents", "vlr_rating", "acs", "kills", "deaths", "assists"]
        )
        scoped_keys.append(f"legacy:{match_id}:{team_identity}:{player}:{signature}")

    output["_row_key"] = scoped_keys
    output = output.drop_duplicates(subset=["_row_key"], keep="last").drop(columns=["_row_key"])
    output["_date_sort"] = pd.to_datetime(output["match_date"], utc=True, errors="coerce")
    output = output.sort_values(
        ["_date_sort", "match_id", "map_number", "team", "player"],
        na_position="first",
    ).drop(columns=["_date_sort"])
    return output.reindex(columns=MATCH_COLUMNS).reset_index(drop=True)


def merge_match_history(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    team_pages: dict[str, str] | None = None,
) -> pd.DataFrame:
    old = normalize_match_dataframe(existing, team_pages)
    new = dedupe_match_rows(normalize_match_dataframe(incoming, team_pages))
    if new.empty:
        return dedupe_match_rows(old)

    refreshed_ids = set(new["match_id"].dropna().astype(int))
    old = old[~old["match_id"].isin(refreshed_ids)].copy()
    return dedupe_match_rows(pd.concat([old, new], ignore_index=True))


def archive_excluded_match_rows(
    excluded: pd.DataFrame,
    archive_csv: str | Path = EXCLUDED_MATCHES_CSV,
    team_pages: dict[str, str] | None = None,
) -> int:
    if excluded.empty:
        return 0
    try:
        archived_raw = pd.read_csv(archive_csv, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        archived_raw = pd.DataFrame(columns=MATCH_COLUMNS)
    archived = merge_match_history(archived_raw, excluded, team_pages)
    Path(archive_csv).parent.mkdir(parents=True, exist_ok=True)
    archived.to_csv(archive_csv, index=False)
    return _unique_match_count(excluded)


def complete_match_ids(matches: pd.DataFrame) -> set[int]:
    if matches.empty:
        return set()
    scoped = matches[
        matches["map_id"].notna()
        & (matches["stat_scope"].astype(str).str.lower() == "map")
    ].copy()
    if scoped.empty:
        return set()
    scoped["_player_id_known"] = pd.to_numeric(
        scoped["player_id"],
        errors="coerce",
    ).notna()
    scoped["_event_known"] = (
        scoped["event_name"].fillna("").astype(str).str.strip().ne("")
    )
    scoped["_team_score_known"] = pd.to_numeric(
        scoped["map_team_score"],
        errors="coerce",
    ).notna()
    scoped["_opponent_score_known"] = pd.to_numeric(
        scoped["map_opp_score"],
        errors="coerce",
    ).notna()
    summary = scoped.groupby("match_id", sort=False).agg(
        rows=("match_id", "size"),
        player_id_coverage=("_player_id_known", "mean"),
        event_known=("_event_known", "max"),
        team_score_known=("_team_score_known", "max"),
        opponent_score_known=("_opponent_score_known", "max"),
    )
    complete = summary[
        summary["rows"].ge(10)
        & summary["player_id_coverage"].ge(0.8)
        & summary["event_known"]
        & summary["team_score_known"]
        & summary["opponent_score_known"]
    ]
    return {int(match_id) for match_id in complete.index}


def _unique_match_count(matches: pd.DataFrame) -> int:
    if matches.empty:
        return 0
    with_ids = int(matches["match_id"].dropna().nunique()) if "match_id" in matches else 0
    without_ids = (
        matches.loc[matches["match_id"].isna(), "match_url"].nunique()
        if {"match_id", "match_url"}.issubset(matches.columns)
        else 0
    )
    return with_ids + int(without_ids)


def match_coverage_report(
    matches: pd.DataFrame,
    team_pages: dict[str, str],
    min_matches_per_team: int = 20,
    candidate_counts: dict[str, int] | None = None,
    season_year: int | None = None,
) -> pd.DataFrame:
    candidate_counts = candidate_counts or {}
    matches = normalize_match_dataframe(matches, team_pages)
    if not matches.empty:
        matches["_date_sort"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce")
    rows = []

    for team, team_url in team_pages.items():
        all_team_matches = pd.DataFrame()
        if not matches.empty and {"team", "match_url"}.issubset(matches.columns):
            all_team_matches = matches[matches["team"] == team].copy()
        ready_team_matches = all_team_matches[
            all_team_matches["map_id"].notna()
            & (all_team_matches["stat_scope"].astype(str).str.lower() == "map")
        ].copy() if not all_team_matches.empty else all_team_matches
        team_matches = ready_team_matches
        if season_year is not None and not team_matches.empty:
            team_matches = team_matches[team_matches["_date_sort"].dt.year == season_year].copy()
        match_count = _unique_match_count(team_matches)
        total_match_count = _unique_match_count(all_team_matches)
        legacy_match_count = max(total_match_count - _unique_match_count(ready_team_matches), 0)
        row_count = int(len(team_matches))
        has_url = bool(str(team_url).strip())
        missing = max(min_matches_per_team - match_count, 0)
        if not has_url:
            status = "missing_url"
        elif match_count >= min_matches_per_team:
            status = "ok"
        elif match_count > 0:
            status = "under_target"
        else:
            status = "no_rows"

        rows.append(
            {
                "team": team,
                "has_url": has_url,
                "candidate_urls": int(candidate_counts.get(team, 0)),
                "matches": match_count,
                "matches_total": total_match_count,
                "legacy_matches": legacy_match_count,
                "season_year": season_year,
                "rows": row_count,
                "target_matches": min_matches_per_team,
                "missing_matches": missing,
                "status": status,
            }
        )

    if not rows:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS).sort_values(["status", "matches", "team"]).reset_index(drop=True)


def team_profile_url_from_matches_url(team_url: str) -> str:
    return team_url.replace("/team/matches/", "/team/")


def _player_id_from_href(href: str) -> int | None:
    match = re.search(r"/player/(\d+)/", href or "")
    return int(match.group(1)) if match else None


def _team_id_from_href(href: str) -> int | None:
    match = re.search(r"/team/(\d+)/", href or "")
    return int(match.group(1)) if match else None


def parse_team_roster_page(session: requests.Session, team_name: str, team_url: str) -> list[dict]:
    profile_url = team_profile_url_from_matches_url(team_url)
    soup = get_soup(session, profile_url)
    rows = []

    for item in soup.select(".team-roster-item"):
        link = item.select_one("a[href*='/player/']")
        alias_node = item.select_one(".team-roster-item-name-alias")
        if not link or not alias_node:
            continue

        href = link.get("href", "")
        player = alias_node.get_text(" ", strip=True)
        role_node = item.select_one(".team-roster-item-name-role")
        role = role_node.get_text(" ", strip=True) if role_node else "active"
        role_norm = role.lower()
        is_staff = "coach" in role_norm or "manager" in role_norm or "analyst" in role_norm
        status = "active" if role_norm == "active" else role_norm
        is_active_player = not is_staff and status not in {"inactive", "reserve", "bench"}

        real_name_node = item.select_one(".team-roster-item-name-real")
        real_name = real_name_node.get_text(" ", strip=True) if real_name_node else ""

        rows.append(
            {
                "team": team_name,
                "player": player,
                "player_id": _player_id_from_href(href),
                "player_url": absolute_url(href),
                "real_name": real_name,
                "roster_role": role,
                "status": status,
                "is_active_player": is_active_player,
                "is_staff": is_staff,
                "team_url": profile_url,
                "scraped_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

    return rows


def scrape_rosters(
    output_csv: str = ROSTERS_CSV,
    team_pages: dict[str, str] | None = None,
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    session = make_session()
    pages = team_pages or TEAM_PAGES
    records = []
    refreshed_teams = set()
    failed_teams = []
    for team_name, team_url in pages.items():
        print(f"Collecting roster for {team_name}")
        try:
            parsed = parse_team_roster_page(session, team_name, team_url)
            if parsed:
                records.extend(parsed)
                refreshed_teams.add(team_name)
            else:
                failed_teams.append(team_name)
                print(f"Keeping the previous roster for {team_name}: no roster rows found")
        except requests.RequestException as exc:
            failed_teams.append(team_name)
            print(f"Skipping roster for {team_name}: {exc}")
        time.sleep(pause_seconds)

    incoming = pd.DataFrame(records)
    try:
        existing = pd.read_csv(output_csv, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame()
    if not existing.empty and "team" in existing.columns:
        existing = existing[~existing["team"].astype(str).isin(refreshed_teams)]
    df = pd.concat([existing, incoming], ignore_index=True, sort=False)
    dedupe_columns = [
        column for column in ["team", "player_id", "player"] if column in df.columns
    ]
    if dedupe_columns:
        df = df.drop_duplicates(subset=dedupe_columns, keep="last")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    from .storage import sync_dataframe

    sync_dataframe("rosters", df, source=str(output_csv))
    df.attrs["scrape_summary"] = {
        "requested_teams": len(pages),
        "refreshed_teams": len(refreshed_teams),
        "failed_teams": failed_teams,
        "roster_rows": len(df),
    }
    return df


def _score_from_header(soup: BeautifulSoup) -> tuple[int | None, int | None]:
    score_box = soup.select_one(".match-header-vs-score")
    if not score_box:
        return None, None

    nums = [int(x) for x in re.findall(r"\b\d+\b", score_box.get_text(" ", strip=True))]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None


def _first_number(cell) -> float | None:
    preferred = cell.select_one(".side.mod-both")
    if preferred:
        text = preferred.get_text(strip=True)
    else:
        text = cell.get_text(" ", strip=True)

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _parse_player_row(
    row,
    teams: tuple[str, str],
    team_ids: tuple[int | None, int | None],
    url: str,
    winner: str | None,
    winner_id: int | None,
    row_team: str | None = None,
) -> dict | None:
    cols = row.find_all("td")
    if len(cols) < 5:
        return None

    try:
        player_link = row.select_one("td.mod-player a[href*='/player/']")
        if player_link:
            player_name = player_link.select_one(".text-of")
            player = player_name.get_text(strip=True) if player_name else list(player_link.stripped_strings)[0]
            player_id = _player_id_from_href(player_link.get("href", ""))
        else:
            parts = list(cols[0].stripped_strings)
            player = parts[0]
            player_id = None

        if row_team not in teams:
            return None

        row_team_index = 0 if row_team == teams[0] else 1
        opponent_index = 1 - row_team_index
        opponent = teams[opponent_index]

        agents = []
        for image in row.select("td.mod-agents img[title]"):
            agents.append(image["title"].strip())

        stat_cells = row.select("td.mod-stat")
        if len(stat_cells) < 5:
            return None

        vlr_rating = _first_number(stat_cells[0])
        acs = _first_number(stat_cells[1])
        kills = _first_number(stat_cells[2])
        deaths = _first_number(stat_cells[3])
        assists = _first_number(stat_cells[4])
        if acs is None or kills is None or deaths is None:
            return None

        return {
            "match_url": url,
            "team": row_team,
            "team_id": team_ids[row_team_index],
            "opponent": opponent,
            "opponent_id": team_ids[opponent_index],
            "winner": winner,
            "winner_id": winner_id,
            "player": player,
            "player_id": player_id,
            "agents": ";".join(agents),
            "vlr_rating": vlr_rating,
            "acs": acs,
            "kills": int(kills),
            "deaths": int(deaths),
            "assists": int(assists or 0),
            "is_winner": row_team == winner if winner else None,
            "scraped_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    except (IndexError, TypeError, ValueError):
        return None


def _parse_overview_player_row(
    row,
    teams: tuple[str, str],
    team_ids: tuple[int | None, int | None],
    url: str,
    winner: str | None,
    winner_id: int | None,
    row_team: str | None = None,
) -> dict | None:
    try:
        player_link = row.select_one(".ovw-cell.mod-player a[href*='/player/']")
        if player_link is None or row_team not in teams:
            return None
        player_name = player_link.select_one(".ovw-player-name")
        player = (
            player_name.get_text(strip=True)
            if player_name
            else list(player_link.stripped_strings)[0]
        )
        player_id = _player_id_from_href(player_link.get("href", ""))
        row_team_index = 0 if row_team == teams[0] else 1
        opponent_index = 1 - row_team_index

        rating_cell = row.select_one(".ovw-cell[data-col='rating2']")
        acs_cell = row.select_one(".ovw-cell[data-col='acs']")
        kills_cell = row.select_one(".ovw-kda-stat[data-col='kills']")
        deaths_cell = row.select_one(".ovw-kda-stat[data-col='deaths']")
        assists_cell = row.select_one(".ovw-kda-stat[data-col='assists']")
        if not all([rating_cell, acs_cell, kills_cell, deaths_cell, assists_cell]):
            return None

        vlr_rating = _first_number(rating_cell)
        acs = _first_number(acs_cell)
        kills = _first_number(kills_cell)
        deaths = _first_number(deaths_cell)
        assists = _first_number(assists_cell)
        if acs is None or kills is None or deaths is None:
            return None

        agents = [
            str(image.get("title") or image.get("alt") or "").strip()
            for image in row.select(".ovw-agents img")
            if str(image.get("title") or image.get("alt") or "").strip()
        ]
        return {
            "match_url": url,
            "team": row_team,
            "team_id": team_ids[row_team_index],
            "opponent": teams[opponent_index],
            "opponent_id": team_ids[opponent_index],
            "winner": winner,
            "winner_id": winner_id,
            "player": player,
            "player_id": player_id,
            "agents": ";".join(agents),
            "vlr_rating": vlr_rating,
            "acs": acs,
            "kills": int(kills),
            "deaths": int(deaths),
            "assists": int(assists or 0),
            "is_winner": row_team == winner if winner else None,
            "scraped_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    except (IndexError, TypeError, ValueError):
        return None


def parse_match_page(session: requests.Session, url: str) -> list[dict]:
    soup = get_soup(session, url)
    team_nodes = soup.select(".match-header .wf-title-med")
    if len(team_nodes) != 2:
        return []

    team_names = [node.get_text(strip=True) for node in team_nodes]
    team_ids = []
    for node in team_nodes:
        link = node.find_parent("a", href=True)
        team_ids.append(_team_id_from_href(link.get("href", "")) if link else None)

    teams = (team_names[0], team_names[1])
    ids = (team_ids[0], team_ids[1])
    url = canonical_match_url(url)
    match_id = match_id_from_url(url)
    match_date = match_date_from_page(soup)
    event_link = soup.select_one(".match-header-event a[href*='/event/']")
    if event_link is None:
        event_link = soup.select_one(".match-header-super a[href*='/event/']")
    event_name = event_link.get_text(" ", strip=True) if event_link else ""
    event_id = None
    if event_link:
        event_match = re.search(r"/event/(\d+)/", event_link.get("href", ""))
        event_id = int(event_match.group(1)) if event_match else None
    event_series_node = soup.select_one(".match-header-event-series")
    event_series = event_series_node.get_text(" ", strip=True) if event_series_node else ""
    if event_series and event_name.endswith(event_series):
        event_name = event_name[: -len(event_series)].rstrip(" :")
    event_stage = event_series.split(":", 1)[-1].strip() if event_series else ""
    event_tier = infer_event_tier(event_name)
    event_region = infer_event_region(event_name, event_series)
    is_lan = infer_is_lan(event_name)
    match_importance = infer_match_importance(event_name, event_stage)
    veto_node = soup.select_one(".match-header-note")
    map_veto = veto_node.get_text(" ", strip=True) if veto_node else ""
    veto_maps = map_metadata_from_veto(map_veto, teams)
    score1, score2 = _score_from_header(soup)
    patch = estimated_patch_epoch(match_date)
    winner = None
    winner_id = None
    if score1 is not None and score2 is not None and score1 != score2:
        winner = teams[0] if score1 > score2 else teams[1]
        winner_id = ids[0] if score1 > score2 else ids[1]

    records = []
    map_number = 0
    for game in soup.select(".vm-stats-game[data-game-id]"):
        map_id_text = str(game.get("data-game-id", "")).strip()
        if not map_id_text.isdigit():
            continue

        tables = game.select("table.wf-table-inset")
        row_groups = []
        if len(tables) >= 2:
            row_groups = [
                (teams[table_index], table.select("tbody tr"), _parse_player_row)
                for table_index, table in enumerate(tables[:2])
            ]
        else:
            overview_rows = [
                row
                for row in game.select(".ovw-table .ovw-row")
                if row.select_one(".ovw-cell.mod-player a[href*='/player/']")
            ]
            if len(overview_rows) >= 10 and len(overview_rows) % 2 == 0:
                midpoint = len(overview_rows) // 2
                row_groups = [
                    (teams[0], overview_rows[:midpoint], _parse_overview_player_row),
                    (teams[1], overview_rows[midpoint:], _parse_overview_player_row),
                ]
        if len(row_groups) < 2:
            continue

        map_number += 1
        map_id = int(map_id_text)
        map_name_node = game.select_one(".vm-stats-game-header .map span")
        map_name = normalize_map_name(
            map_name_node.get_text(" ", strip=True) if map_name_node else ""
        )
        veto_metadata = veto_maps.get(team_key(map_name), {})
        pick_node = game.select_one(".vm-stats-game-header .map .picked, .vm-stats-game-header .map .pick")
        pick_type = veto_metadata.get("map_pick_type")
        if not pick_type and pick_node is not None:
            pick_type = "pick"
        map_scores = [
            _first_number(node)
            for node in game.select(".vm-stats-game-header .team .score")[:2]
        ]
        map_score1 = int(map_scores[0]) if len(map_scores) > 0 and map_scores[0] is not None else None
        map_score2 = int(map_scores[1]) if len(map_scores) > 1 and map_scores[1] is not None else None

        for row_team, player_rows, row_parser in row_groups:
            for row in player_rows:
                parsed = row_parser(
                    row,
                    teams,
                    ids,
                    url,
                    winner,
                    winner_id,
                    row_team=row_team,
                )
                if parsed:
                    is_first_team = parsed["team"] == teams[0]
                    parsed.update(
                        {
                            "match_id": match_id,
                            "match_date": match_date,
                            "season_year": pd.to_datetime(
                                match_date, utc=True, errors="coerce"
                            ).year
                            if pd.notna(pd.to_datetime(match_date, utc=True, errors="coerce"))
                            else None,
                            "event_id": event_id,
                            "event_name": event_name,
                            "event_series": event_series,
                            "event_stage": event_stage,
                            "event_tier": event_tier,
                            "event_region": event_region,
                            "is_lan": is_lan,
                            "patch": patch,
                            "patch_source": "date_epoch" if patch else "",
                            "map_veto": map_veto,
                            "match_importance": match_importance,
                            "competition_tier": event_tier,
                            "competition_strength_weight": 1.0,
                            "team_tier_at_match": event_tier,
                            "opponent_tier_at_match": event_tier,
                            "team_promoted_next_season": False,
                            "opponent_promoted_next_season": False,
                            "data_source": "vlr_html",
                            "map_id": map_id,
                            "map_number": map_number,
                            "map_name": map_name,
                            "map_pick_team": veto_metadata.get("map_pick_team", ""),
                            "map_pick_type": pick_type or "unknown",
                            "map_veto_order": veto_metadata.get("map_veto_order"),
                            "map_data_source": "vlr_html",
                            "stat_scope": "map",
                            "team_score": score1 if is_first_team else score2,
                            "opp_score": score2 if is_first_team else score1,
                            "team_region": "",
                            "opponent_region": "",
                            "map_team_score": map_score1 if is_first_team else map_score2,
                            "map_opp_score": map_score2 if is_first_team else map_score1,
                        }
                    )
                    records.append(parsed)
    return records


def _upcoming_match_date(card) -> str | None:
    utc_node = card.select_one("[data-utc-ts]")
    if utc_node and utc_node.get("data-utc-ts"):
        raw = str(utc_node.get("data-utc-ts")).strip()
        if raw.isdigit():
            unit = "ms" if len(raw) >= 13 else "s"
            parsed = pd.to_datetime(int(raw), unit=unit, utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.isoformat()

    label = card.find_previous(class_=lambda value: value and "wf-label" in value and "mod-large" in value)
    time_node = card.select_one(".match-item-time")
    if not label or not time_node:
        return None
    date_text = re.sub(r"\b(today|tomorrow)\b", "", label.get_text(" ", strip=True), flags=re.I).strip(" ,")
    if not re.search(r"\b20\d{2}\b", date_text):
        date_text = f"{date_text}, {datetime.utcnow().year}"
    parsed = pd.to_datetime(
        f"{date_text} {time_node.get_text(' ', strip=True)}",
        utc=True,
        errors="coerce",
    )
    return parsed.isoformat() if pd.notna(parsed) else None


def parse_upcoming_matches_page(
    soup: BeautifulSoup,
    team_pages: dict[str, str],
) -> pd.DataFrame:
    lookup = canonical_team_lookup(team_pages)
    active_teams = set(team_pages)
    rows = []
    cards = soup.select("a.wf-module-item.match-item[href]")
    if not cards:
        cards = soup.select("a.match-item[href]")

    for card in cards:
        href = card.get("href", "")
        match_url = canonical_match_url(href)
        match_id = match_id_from_url(match_url)
        team_nodes = card.select(".match-item-vs-team-name")
        if len(team_nodes) < 2:
            continue
        team1 = canonical_team_name(team_nodes[0].get_text(" ", strip=True), lookup)
        team2 = canonical_team_name(team_nodes[1].get_text(" ", strip=True), lookup)
        if team1 not in active_teams or team2 not in active_teams or team1 == team2:
            continue

        card_text = card.get_text(" ", strip=True)
        if "live" in card_text.lower():
            continue
        match_date = _upcoming_match_date(card)
        event_node = card.select_one(".match-item-event-name")
        event_series_node = card.select_one(".match-item-event-series")
        event_name = event_node.get_text(" ", strip=True) if event_node else ""
        event_series = event_series_node.get_text(" ", strip=True) if event_series_node else ""
        event_stage = event_series.split(":", 1)[-1].strip() if event_series else ""
        best_of_match = re.search(r"\bbo([135])\b", card_text, flags=re.I)
        rows.append(
            {
                "match_id": match_id,
                "match_url": match_url,
                "match_date": match_date,
                "team1": team1,
                "team2": team2,
                "event_name": event_name,
                "event_series": event_series,
                "event_stage": event_stage,
                "event_tier": infer_event_tier(event_name),
                "is_lan": infer_is_lan(event_name),
                "best_of": int(best_of_match.group(1)) if best_of_match else 3,
                "scraped_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    output = pd.DataFrame(rows, columns=UPCOMING_COLUMNS)
    output["_date"] = pd.to_datetime(output["match_date"], utc=True, errors="coerce")
    output = output.drop_duplicates(subset=["match_id"], keep="last")
    return output.sort_values(["_date", "match_id"], na_position="last").drop(columns=["_date"]).reset_index(drop=True)


def scrape_upcoming_matches(
    output_csv: str = UPCOMING_MATCHES_CSV,
    team_pages: dict[str, str] | None = None,
) -> pd.DataFrame:
    pages = team_pages or TEAM_PAGES
    soup = get_soup(make_session(), f"{BASE_URL}/matches")
    upcoming = parse_upcoming_matches_page(soup, pages)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    upcoming.to_csv(output_csv, index=False)
    from .storage import sync_dataframe

    sync_dataframe("upcoming_matches", upcoming, source=str(output_csv))
    return upcoming


def enrich_incomplete_match_history(
    output_csv: str = MATCHES_CSV,
    team_pages: dict[str, str] | None = None,
    season_year: int | None = None,
    max_matches: int | None = 100,
    pause_seconds: float = 0.35,
    progress=None,
    vlrggapi_base_url: str = VLRGGAPI_BASE_URL,
    use_vlrggapi_enrichment: bool = False,
) -> tuple[pd.DataFrame, dict]:
    pages = team_pages or TEAM_PAGES
    try:
        existing_raw = pd.read_csv(output_csv, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=MATCH_COLUMNS), {"candidates": 0, "refreshed": 0, "failed": 0}

    existing = normalize_match_dataframe(existing_raw, pages)
    if existing.empty:
        return existing, {"candidates": 0, "refreshed": 0, "failed": 0}

    candidate_rows = existing[
        existing["map_id"].notna()
        & existing["stat_scope"].fillna("").astype(str).str.lower().eq("map")
    ].copy()
    candidate_rows["_date"] = pd.to_datetime(
        candidate_rows["match_date"],
        utc=True,
        errors="coerce",
    )
    candidate_rows["_player_id_known"] = pd.to_numeric(
        candidate_rows["player_id"],
        errors="coerce",
    ).notna()
    candidate_rows["_event_known"] = (
        candidate_rows["event_name"].fillna("").astype(str).str.strip().ne("")
    )
    candidate_rows["_team_score_known"] = pd.to_numeric(
        candidate_rows["map_team_score"],
        errors="coerce",
    ).notna()
    candidate_rows["_opponent_score_known"] = pd.to_numeric(
        candidate_rows["map_opp_score"],
        errors="coerce",
    ).notna()
    candidate_summary = candidate_rows.groupby("match_id", sort=False).agg(
        latest_date=("_date", "max"),
        rows=("match_id", "size"),
        player_id_coverage=("_player_id_known", "mean"),
        event_known=("_event_known", "max"),
        team_score_known=("_team_score_known", "max"),
        opponent_score_known=("_opponent_score_known", "max"),
        match_url=("match_url", "first"),
    )
    if season_year is not None:
        candidate_summary = candidate_summary[
            candidate_summary["latest_date"].dt.year.eq(int(season_year))
        ]
    candidate_summary = candidate_summary[
        candidate_summary["rows"].lt(10)
        | candidate_summary["player_id_coverage"].lt(0.8)
        | ~candidate_summary["event_known"]
        | ~candidate_summary["team_score_known"]
        | ~candidate_summary["opponent_score_known"]
    ]
    candidates = [
        (row.latest_date, int(match_id), canonical_match_url(row.match_url))
        for match_id, row in candidate_summary.iterrows()
    ]

    oldest = pd.Timestamp("1900-01-01", tz="UTC")
    candidates.sort(
        key=lambda item: (oldest if pd.isna(item[0]) else item[0], item[1]),
        reverse=True,
    )
    total_candidates = len(candidates)
    if max_matches is not None:
        candidates = candidates[: max(0, int(max_matches))]

    session = make_session()
    api_available = bool(
        use_vlrggapi_enrichment
        and vlrggapi_base_url
        and vlrggapi_is_healthy(session, vlrggapi_base_url)
    )
    records = []
    failures = []
    for index, (_, match_id, url) in enumerate(candidates, start=1):
        if progress:
            progress("metadata_backfill", index - 1, len(candidates), f"Enriching match {match_id}")
        try:
            parsed = parse_match_page(session, url)
            if parsed and api_available:
                try:
                    metadata = fetch_vlrggapi_match_metadata(
                        session,
                        vlrggapi_base_url,
                        match_id,
                    )
                    parsed = enrich_match_records_with_map_metadata(parsed, metadata)
                except (requests.RequestException, TypeError, ValueError):
                    pass
            if parsed:
                records.extend(parsed)
            else:
                failures.append({"match_id": match_id, "error": "No player-map rows"})
        except (requests.RequestException, TypeError, ValueError) as exc:
            failures.append({"match_id": match_id, "error": str(exc)})
        if pause_seconds:
            time.sleep(pause_seconds)

    incoming = pd.DataFrame(canonicalize_match_records(records, pages), columns=MATCH_COLUMNS)
    incoming = normalize_match_dataframe(incoming, pages)
    merged = merge_match_history(existing, incoming, pages)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    from .storage import sync_match_data

    sync_match_data(merged, source=str(output_csv))
    if progress:
        progress("metadata_backfill", len(candidates), len(candidates), "Match metadata enrichment complete")
    refreshed = int(incoming["match_id"].nunique()) if not incoming.empty else 0
    summary = {
        "candidates": total_candidates,
        "attempted": len(candidates),
        "refreshed": refreshed,
        "failed": len(failures),
        "remaining": max(0, total_candidates - refreshed),
        "failures": failures[:20],
    }
    return merged, summary


def scrape_matches(
    output_csv: str = MATCHES_CSV,
    limit_per_team: int | None = None,
    team_pages: dict[str, str] | None = None,
    pause_seconds: float = 1.2,
    season_year: int | None = None,
    recent_days: int | None = None,
    min_matches_per_team: int = 20,
    coverage_csv: str = MATCH_COVERAGE_CSV,
    save_filtered: bool = False,
    preserve_history: bool = True,
    refresh_existing: bool = False,
    vlrggapi_base_url: str = VLRGGAPI_BASE_URL,
    use_vlrggapi_enrichment: bool = False,
    excluded_csv: str | Path = EXCLUDED_MATCHES_CSV,
) -> pd.DataFrame:
    session = make_session()
    pages = team_pages or TEAM_PAGES
    vlrggapi_available = bool(
        use_vlrggapi_enrichment
        and vlrggapi_base_url
        and vlrggapi_is_healthy(session, vlrggapi_base_url)
    )
    if use_vlrggapi_enrichment and vlrggapi_base_url and not vlrggapi_available:
        print(
            f"VLRGGAPI enrichment is unavailable at {vlrggapi_base_url}; "
            "continuing with direct VLR parsing."
        )
    try:
        existing_raw = pd.read_csv(output_csv) if preserve_history else pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing_raw = pd.DataFrame()
    existing = normalize_match_dataframe(existing_raw, pages)
    reusable_match_ids = complete_match_ids(existing) if not refresh_existing else set()

    match_urls_by_id = {}
    candidate_counts = {}
    candidate_limit = limit_per_team if limit_per_team and limit_per_team > 0 else None
    discovered_match_ids = set()
    rejection_counts = {
        "wrong_season": 0,
        "missing_date": 0,
        "explicit_non_tier1_event": 0,
        "non_tier1_matchup": 0,
    }

    for team_name, team_url in pages.items():
        print(f"Collecting matches for {team_name}")
        try:
            discovered = get_team_match_candidates(session, team_url)
        except requests.RequestException as exc:
            print(f"Skipping match list for {team_name}: {exc}")
            candidate_counts[team_name] = 0
            time.sleep(pause_seconds)
            continue
        discovered_match_ids.update(
            candidate["match_id"] for candidate in discovered if candidate.get("match_id") is not None
        )
        relevant, rejected = filter_tier1_match_candidates(
            discovered,
            season_year,
            pages.keys(),
        )
        for reason, count in rejected.items():
            rejection_counts[reason] += count
        if candidate_limit is not None:
            relevant = relevant[:candidate_limit]
        candidate_counts[team_name] = len(relevant)
        for candidate in relevant:
            match_urls_by_id.setdefault(candidate["match_id"], candidate["match_url"])
        time.sleep(pause_seconds)

    records = []
    parsed_match_ids = set()
    failed_match_ids = set()
    skipped_existing = 0
    match_items = list(match_urls_by_id.items())
    for index, (match_id, url) in enumerate(match_items, 1):
        if match_id in reusable_match_ids:
            skipped_existing += 1
            continue

        print(f"[{index}/{len(match_items)}] {url}")
        try:
            parsed = parse_match_page(session, url)
            if parsed and vlrggapi_available:
                try:
                    metadata = fetch_vlrggapi_match_metadata(
                        session,
                        vlrggapi_base_url,
                        match_id,
                    )
                    parsed = enrich_match_records_with_map_metadata(parsed, metadata)
                except (requests.RequestException, TypeError, ValueError) as exc:
                    print(f"VLRGGAPI enrichment skipped for {match_id}: {exc}")
            if not parsed:
                failed_match_ids.add(match_id)
                print(f"Skipping {url}: no map-level player stats found")
            else:
                records.extend(parsed)
                parsed_match_ids.add(match_id)
        except (requests.RequestException, TypeError, ValueError) as exc:
            failed_match_ids.add(match_id)
            print(f"Skipping {url}: {exc}")
        time.sleep(pause_seconds)

    incoming = pd.DataFrame(canonicalize_match_records(records, pages), columns=MATCH_COLUMNS)
    incoming = normalize_match_dataframe(incoming, pages)
    if preserve_history:
        output_df = merge_match_history(existing, incoming, pages)
    else:
        output_df = dedupe_match_rows(incoming)

    from .team_registry import load_team_registry

    registry = load_team_registry()
    if season_year is not None:
        supplemental_registry = pd.DataFrame(
            {
                "team": list(pages),
                "tier": "tier1",
                "active": True,
                "season_year": int(season_year),
            }
        )
        registry = pd.concat([registry, supplemental_registry], ignore_index=True)
    output_df, excluded = partition_registry_tier1_history(output_df, registry)
    excluded_matches = archive_excluded_match_rows(excluded, excluded_csv, pages)

    if save_filtered and not output_df.empty and (season_year is not None or recent_days is not None):
        output_df = filter_matches_by_time(output_df, season_year=season_year, recent_days=recent_days)
        output_df = output_df.reindex(columns=MATCH_COLUMNS)
        output_df = dedupe_match_rows(output_df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    from .storage import sync_match_data

    sync_match_data(output_df, source=str(output_csv))

    coverage = match_coverage_report(
        output_df,
        pages,
        min_matches_per_team=min_matches_per_team,
        candidate_counts=candidate_counts,
        season_year=season_year,
    )
    Path(coverage_csv).parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_csv, index=False)
    below_target = coverage[coverage["status"] != "ok"]
    if not below_target.empty:
        print(
            f"Coverage warning: {len(below_target)} of {len(coverage)} teams have fewer than "
            f"{min_matches_per_team} parsed matches. See {coverage_csv}."
        )
    output_df.attrs["scrape_summary"] = {
        "discovered_matches": len(discovered_match_ids),
        "relevant_matches": len(match_items),
        "filtered_candidate_cards": sum(rejection_counts.values()),
        "candidate_rejections": rejection_counts,
        "archived_non_tier1_matches": excluded_matches,
        "reused_matches": skipped_existing,
        "parsed_matches": len(parsed_match_ids),
        "failed_matches": len(failed_match_ids),
        "new_player_map_rows": len(incoming),
        "total_player_map_rows": len(output_df),
        "total_matches": _unique_match_count(output_df),
        "season_year": season_year,
    }
    return output_df


def _looks_like_news_url(href: str) -> bool:
    return bool(re.match(r"^/\d+/", href or ""))


def _extract_news_date(parts: Iterable[str]) -> str | None:
    date_pattern = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")
    for part in parts:
        clean = part.strip()
        if date_pattern.match(clean):
            return clean
    return None


def parse_news_index_page(session: requests.Session, page: int) -> list[dict]:
    url = f"{BASE_URL}/news" if page == 1 else f"{BASE_URL}/news/?page={page}"
    soup = get_soup(session, url)
    items = []

    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if not _looks_like_news_url(href):
            continue

        parts = [p.strip() for p in link.get_text("\n", strip=True).split("\n") if p.strip()]
        if not parts:
            continue

        title = parts[0]
        published = _extract_news_date(parts)
        author = next((p.replace("by ", "").strip() for p in parts if p.startswith("by ")), None)
        summary_parts = [
            p
            for p in parts[1:]
            if p != published and not p.startswith("by ") and p not in {"VLR.gg", "News", "•"}
        ]

        items.append(
            {
                "url": absolute_url(href),
                "title": title,
                "summary": " ".join(summary_parts),
                "published": published,
                "author": author,
                "source_page": url,
                "scraped_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

    seen = set()
    deduped = []
    for item in items:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        deduped.append(item)
    return deduped


def scrape_news(
    output_csv: str = NEWS_CSV,
    pages: int = 3,
    pause_seconds: float = 1.0,
    preserve_history: bool = True,
) -> pd.DataFrame:
    session = make_session()
    records = []
    for page in range(1, pages + 1):
        print(f"Collecting VLR news page {page}")
        records.extend(parse_news_index_page(session, page))
        time.sleep(pause_seconds)

    incoming = pd.DataFrame(records)
    try:
        existing = pd.read_csv(output_csv) if preserve_history else pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame()
    df = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if not df.empty and "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="last")
        df["_published"] = pd.to_datetime(df.get("published"), utc=True, errors="coerce")
        df = df.sort_values(["_published", "url"], ascending=[False, True], na_position="last")
        df = df.drop(columns=["_published"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    from .features.news_impact import structure_news_events
    from .storage import sync_dataframe

    events = structure_news_events(df)
    events.to_csv(NEWS_EVENTS_CSV, index=False)
    sync_dataframe("news", df, source=str(output_csv))
    sync_dataframe("news_events", events, source=str(NEWS_EVENTS_CSV))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape VLR match stats and news.")
    parser.add_argument("--matches", action="store_true", help="Scrape match/player stats.")
    parser.add_argument("--news", action="store_true", help="Scrape VLR news.")
    parser.add_argument("--rosters", action="store_true", help="Scrape current VLR team rosters.")
    parser.add_argument("--upcoming", action="store_true", help="Scrape upcoming Tier 1 matches from VLR.")
    parser.add_argument(
        "--enrich-history",
        action="store_true",
        help="Refresh stored matches missing essential player, event, or map-score data.",
    )
    parser.add_argument(
        "--limit-per-team",
        type=int,
        default=0,
        help="Optional discovery cap per team; 0 collects every match card VLR exposes.",
    )
    parser.add_argument("--min-matches-per-team", type=int, default=20)
    parser.add_argument("--news-pages", type=int, default=3)
    parser.add_argument("--max-matches", type=int, default=0, help="Optional cap for --enrich-history; 0 means all candidates.")
    parser.add_argument("--pause-seconds", type=float, default=0.35)
    parser.add_argument("--season-year", type=int, help="Only discover relevant Tier 1 results from this season.")
    parser.add_argument("--recent-days", type=int, help="Optional recency filter when --save-filtered is used.")
    parser.add_argument("--save-filtered", action="store_true", help="Write only the selected year/recency window instead of raw latest matches.")
    parser.add_argument("--replace-history", action="store_true", help="Replace the match CSV instead of merging new matches into it.")
    parser.add_argument("--refresh-existing", action="store_true", help="Re-download matches that already have complete map-level rows.")
    parser.add_argument(
        "--vlrggapi-base-url",
        default=VLRGGAPI_BASE_URL,
        help="Optional self-hosted vlrggapi base URL for map/veto enrichment.",
    )
    parser.add_argument(
        "--use-vlrggapi-enrichment",
        action="store_true",
        help="Opt into expensive helper match-detail enrichment; direct VLR parsing is the default.",
    )
    parser.add_argument("--no-vlrggapi", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    use_vlrggapi_enrichment = args.use_vlrggapi_enrichment and not args.no_vlrggapi

    if not args.matches and not args.news and not args.rosters and not args.upcoming and not args.enrich_history:
        parser.error("Choose --matches, --news, --rosters, --upcoming, --enrich-history, or a combination.")

    if args.matches:
        df = scrape_matches(
            limit_per_team=args.limit_per_team or None,
            season_year=args.season_year,
            recent_days=args.recent_days,
            min_matches_per_team=args.min_matches_per_team,
            save_filtered=args.save_filtered,
            preserve_history=not args.replace_history,
            refresh_existing=args.refresh_existing,
            vlrggapi_base_url=args.vlrggapi_base_url,
            use_vlrggapi_enrichment=use_vlrggapi_enrichment,
        )
        print(f"Saved {len(df)} match stat rows to {MATCHES_CSV}")
    if args.news:
        df = scrape_news(pages=args.news_pages)
        print(f"Saved {len(df)} news rows to {NEWS_CSV}")
    if args.rosters:
        df = scrape_rosters()
        print(f"Saved {len(df)} roster rows to {ROSTERS_CSV}")
    if args.upcoming:
        df = scrape_upcoming_matches()
        print(f"Saved {len(df)} upcoming Tier 1 matches to {UPCOMING_MATCHES_CSV}")
    if args.enrich_history:
        df, summary = enrich_incomplete_match_history(
            season_year=args.season_year,
            max_matches=args.max_matches or None,
            pause_seconds=max(0.0, args.pause_seconds),
            progress=lambda step, current, total, message: print(
                f"[{step}] {current}/{total} {message}",
                flush=True,
            ),
            vlrggapi_base_url=args.vlrggapi_base_url,
            use_vlrggapi_enrichment=use_vlrggapi_enrichment,
        )
        print(json.dumps(summary, indent=2))
        print(f"Saved {len(df)} enriched match stat rows to {MATCHES_CSV}")


if __name__ == "__main__":
    main()
