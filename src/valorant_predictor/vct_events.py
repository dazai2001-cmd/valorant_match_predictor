from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from .config import MATCHES_CSV, VCT_EVENTS_CSV, VLRGGAPI_BASE_URL
from .storage import sync_dataframe, sync_match_data
from .team_registry import (
    VCT_TIER1_SEASON,
    active_tier1_team_names,
    registry_team_pages,
    team_key,
    team_matches_url,
    team_profile_url,
    upsert_season_registry,
)
from .vlr_client import (
    MATCH_COLUMNS,
    canonical_team_id_lookup,
    canonical_team_lookup,
    canonical_team_name,
    canonicalize_match_records,
    complete_match_ids,
    infer_is_lan,
    infer_match_importance,
    make_session,
    merge_match_history,
    normalize_match_dataframe,
    parse_match_page,
)
from .vlrggapi_client import (
    fetch_vlrggapi_event_detail,
    fetch_vlrggapi_event_matches,
    vlrggapi_is_healthy,
)


ProgressCallback = Callable[[str, int, int, str], None]

EVENT_COLUMNS = [
    "event_id",
    "season_year",
    "event_name",
    "league",
    "category",
    "competition_tier",
    "competition_strength_weight",
    "include_training",
    "event_url",
    "team_count",
    "match_count",
    "source",
    "verified_at",
]


VCT_EVENTS = {
    2025: [
        (2274, "VCT 2025: Americas Kickoff", "VCT Americas", "kickoff", "tier1", 1.00),
        (2276, "VCT 2025: EMEA Kickoff", "VCT EMEA", "kickoff", "tier1", 1.00),
        (2277, "VCT 2025: Pacific Kickoff", "VCT Pacific", "kickoff", "tier1", 1.00),
        (2275, "VCT 2025: China Kickoff", "VCT China", "kickoff", "tier1", 1.00),
        (2281, "Valorant Masters Bangkok 2025", "VCT International", "masters", "tier1", 1.05),
        (2347, "VCT 2025: Americas Stage 1", "VCT Americas", "stage_1", "tier1", 1.00),
        (2380, "VCT 2025: EMEA Stage 1", "VCT EMEA", "stage_1", "tier1", 1.00),
        (2379, "VCT 2025: Pacific Stage 1", "VCT Pacific", "stage_1", "tier1", 1.00),
        (2359, "VCT 2025: China Stage 1", "VCT China", "stage_1", "tier1", 1.00),
        (2282, "Valorant Masters Toronto 2025", "VCT International", "masters", "tier1", 1.05),
        (2501, "VCT 2025: Americas Stage 2", "VCT Americas", "stage_2", "tier1", 1.00),
        (2498, "VCT 2025: EMEA Stage 2", "VCT EMEA", "stage_2", "tier1", 1.00),
        (2500, "VCT 2025: Pacific Stage 2", "VCT Pacific", "stage_2", "tier1", 1.00),
        (2499, "VCT 2025: China Stage 2", "VCT China", "stage_2", "tier1", 1.00),
        (2283, "Valorant Champions 2025", "VCT International", "champions", "tier1", 1.08),
        (2534, "VCT 2025: Americas Ascension", "VCT Americas", "ascension", "promotion", 0.55),
        (2519, "VCT 2025: EMEA Ascension", "VCT EMEA", "ascension", "promotion", 0.55),
        (2535, "VCT 2025: Pacific Ascension", "VCT Pacific", "ascension", "promotion", 0.55),
        (2648, "VCT 2025: China Ascension", "VCT China", "ascension", "promotion", 0.55),
    ]
}


def curated_event_rows(season_year: int) -> list[dict]:
    if season_year not in VCT_EVENTS:
        raise ValueError(f"No curated VCT event list is configured for {season_year}.")
    rows = []
    for event_id, name, league, category, tier, weight in VCT_EVENTS[season_year]:
        rows.append(
            {
                "event_id": int(event_id),
                "season_year": int(season_year),
                "event_name": name,
                "league": league,
                "category": category,
                "competition_tier": tier,
                "competition_strength_weight": float(weight),
                "include_training": True,
                "event_url": f"https://www.vlr.gg/event/{event_id}",
                "team_count": 0,
                "match_count": 0,
                "source": "curated_vct_event_id",
                "verified_at": "",
            }
        )
    return rows


def _notify(
    callback: ProgressCallback | None,
    step: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback:
        callback(step, current, total, message)


def _event_team_rows(
    events: list[dict],
    details: dict[int, dict],
    season_year: int,
) -> list[dict]:
    pages = registry_team_pages(season_year=None)
    name_lookup = canonical_team_lookup(pages)
    id_lookup = canonical_team_id_lookup(pages)
    now = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    for event in events:
        if event["competition_tier"] != "tier1" or event["league"] == "VCT International":
            continue
        for team in details.get(int(event["event_id"]), {}).get("teams", []):
            if not isinstance(team, dict):
                continue
            team_id = str(team.get("id", "")).strip()
            raw_name = str(team.get("name", "")).strip()
            if not team_id or not raw_name:
                continue
            canonical = id_lookup.get(team_id) or canonical_team_name(raw_name, name_lookup)
            canonical = str(canonical or raw_name)
            rows.append(
                {
                    "team": canonical,
                    "team_id": team_id,
                    "matches_url": team_matches_url(team_id, canonical),
                    "profile_url": team_profile_url(team_id, canonical),
                    "source": "vlrggapi_vct_event",
                    "league": event["league"],
                    "region": event["league"],
                    "tier": "tier1",
                    "season_year": str(season_year),
                    "active": "true",
                    "ranking_page": event["event_url"],
                    "added_at": now,
                    "resolved_at": now,
                }
            )
    return rows


def load_vct_event_catalog(
    season_year: int,
    session: requests.Session | None = None,
    base_url: str = VLRGGAPI_BASE_URL,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[int, dict], dict[int, list[dict]]]:
    session = session or make_session()
    if not vlrggapi_is_healthy(session, base_url):
        raise ConnectionError(
            f"vlrggapi is not healthy at {base_url}. Start it with docker compose up -d."
        )

    events = curated_event_rows(season_year)
    details: dict[int, dict] = {}
    matches: dict[int, list[dict]] = {}
    now = datetime.utcnow().isoformat(timespec="seconds")
    for index, event in enumerate(events, start=1):
        _notify(
            progress,
            "event_catalog",
            index - 1,
            len(events),
            f"Reading {event['event_name']}",
        )
        event_id = int(event["event_id"])
        detail = fetch_vlrggapi_event_detail(session, base_url, event_id)
        event_matches = fetch_vlrggapi_event_matches(session, base_url, event_id)
        details[event_id] = detail
        matches[event_id] = event_matches
        event["team_count"] = len(detail.get("teams", []))
        event["match_count"] = len(event_matches)
        event["verified_at"] = now

    frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    Path(VCT_EVENTS_CSV).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(VCT_EVENTS_CSV, index=False)
    sync_dataframe("vct_events", frame, source=str(VCT_EVENTS_CSV))

    team_rows = _event_team_rows(events, details, season_year)
    if team_rows:
        upsert_season_registry(team_rows)
    _notify(progress, "event_catalog", len(events), len(events), "VCT event catalog verified")
    return frame, details, matches


def _canonical_event_team(name: str, lookup: dict[str, str]) -> str:
    return str(canonical_team_name(name, lookup) or name).strip()


def _match_metadata(
    events: pd.DataFrame,
    event_matches: dict[int, list[dict]],
    season_year: int,
) -> tuple[dict[int, dict], set[str]]:
    pages = registry_team_pages(season_year=None)
    lookup = canonical_team_lookup(pages)
    current_names = {
        team_key(name) for name in active_tier1_team_names(season_year=season_year)
    }
    try:
        next_names = {
            team_key(name)
            for name in active_tier1_team_names(season_year=season_year + 1)
        }
    except ValueError:
        next_names = set()
    promoted = next_names - current_names
    metadata = {}
    for event in events.to_dict("records"):
        event_id = int(event["event_id"])
        for match in event_matches.get(event_id, []):
            try:
                match_id = int(match.get("match_id"))
            except (TypeError, ValueError):
                continue
            team1 = _canonical_event_team(match.get("team1", {}).get("name", ""), lookup)
            team2 = _canonical_event_team(match.get("team2", {}).get("name", ""), lookup)
            if event["competition_tier"] == "promotion" and not (
                team_key(team1) in promoted or team_key(team2) in promoted
            ):
                continue
            metadata[match_id] = {
                **event,
                "match_id": match_id,
                "match_url": match.get("url", ""),
                "event_series": match.get("event_series", ""),
                "team1": team1,
                "team2": team2,
            }
    return metadata, promoted


def annotate_vct_matches(
    matches: pd.DataFrame,
    metadata: dict[int, dict],
    promoted_team_keys: set[str],
) -> pd.DataFrame:
    if matches.empty or not metadata:
        return matches
    output = matches.copy().astype(object)
    match_ids = pd.to_numeric(output["match_id"], errors="coerce")
    for column in MATCH_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    for match_id, values in metadata.items():
        mask = match_ids.eq(match_id)
        if not mask.any():
            continue
        tier = values["competition_tier"]
        common = {
            "season_year": values["season_year"],
            "event_id": values["event_id"],
            "event_name": values["event_name"],
            "event_series": values.get("event_series", ""),
            "event_stage": values.get("event_series", ""),
            "event_tier": tier,
            "event_region": values.get("league", ""),
            "is_lan": infer_is_lan(values["event_name"]),
            "match_importance": infer_match_importance(
                values["event_name"], values.get("event_series", "")
            ),
            "competition_tier": tier,
            "competition_strength_weight": values["competition_strength_weight"],
            "team_tier_at_match": "tier1" if tier == "tier1" else "tier2",
            "opponent_tier_at_match": "tier1" if tier == "tier1" else "tier2",
            "data_source": "vlrggapi_event+vlr_html",
        }
        for column, value in common.items():
            output.loc[mask, column] = value
        output.loc[mask, "team_promoted_next_season"] = output.loc[mask, "team"].map(
            lambda name: team_key(name) in promoted_team_keys
        )
        output.loc[mask, "opponent_promoted_next_season"] = output.loc[
            mask, "opponent"
        ].map(lambda name: team_key(name) in promoted_team_keys)
    return output.reindex(columns=MATCH_COLUMNS)


def import_vct_season(
    season_year: int = 2025,
    output_csv: str | Path = MATCHES_CSV,
    base_url: str = VLRGGAPI_BASE_URL,
    refresh_existing: bool = False,
    pause_seconds: float = 1.2,
    limit_new_matches: int | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict]:
    session = make_session()
    events, _, event_matches = load_vct_event_catalog(
        season_year,
        session=session,
        base_url=base_url,
        progress=progress,
    )
    metadata, promoted = _match_metadata(events, event_matches, season_year)
    try:
        existing_raw = pd.read_csv(output_csv, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing_raw = pd.DataFrame(columns=MATCH_COLUMNS)
    pages = registry_team_pages(season_year=None)
    existing = normalize_match_dataframe(existing_raw, pages)
    reusable = complete_match_ids(existing) if not refresh_existing else set()
    all_candidates = [
        values
        for match_id, values in sorted(metadata.items())
        if match_id not in reusable
    ]
    candidates = all_candidates
    if limit_new_matches is not None:
        candidates = candidates[: max(0, int(limit_new_matches))]

    records = []
    failures = []
    for index, values in enumerate(candidates, start=1):
        _notify(
            progress,
            "match_import",
            index - 1,
            len(candidates),
            f"Parsing match {values['match_id']}",
        )
        try:
            parsed = parse_match_page(session, values["match_url"])
        except (requests.RequestException, TypeError, ValueError) as exc:
            parsed = []
            failures.append({"match_id": values["match_id"], "error": str(exc)})
        if parsed:
            records.extend(parsed)
        elif not any(item["match_id"] == values["match_id"] for item in failures):
            failures.append({"match_id": values["match_id"], "error": "No player-map rows"})
        if pause_seconds:
            time.sleep(pause_seconds)

    incoming = pd.DataFrame(
        canonicalize_match_records(records, pages),
        columns=MATCH_COLUMNS,
    )
    incoming = normalize_match_dataframe(incoming, pages)
    merged = merge_match_history(existing, incoming, pages)
    merged = annotate_vct_matches(merged, metadata, promoted)
    merged = normalize_match_dataframe(merged, pages)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    sync_match_data(merged, source=str(output_csv))

    summary = {
        "season_year": season_year,
        "events": len(events),
        "official_matches": sum(
            1 for values in metadata.values() if values["competition_tier"] == "tier1"
        ),
        "promotion_matches": sum(
            1 for values in metadata.values() if values["competition_tier"] == "promotion"
        ),
        "reused_matches": len(metadata) - len(all_candidates),
        "missing_matches": len(all_candidates),
        "deferred_matches": len(all_candidates) - len(candidates),
        "parsed_matches": len(set(incoming.get("match_id", pd.Series(dtype=float)).dropna())),
        "failed_matches": len(failures),
        "failures": failures[:20],
        "player_map_rows": len(merged),
        "promoted_team_keys": sorted(promoted),
    }
    _notify(progress, "match_import", len(candidates), len(candidates), "VCT import complete")
    return merged, summary
