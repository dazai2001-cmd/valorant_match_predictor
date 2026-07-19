from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd


PATCH_EPOCH_DAYS = 14
PATCH_EPOCH_ANCHOR = pd.Timestamp("2025-01-07", tz="UTC")
TRAINING_TIERS = {"tier1", "promotion"}


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def identity_key(value) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", _text(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_value.lower())


def classify_event_tier(event_name: str) -> str:
    text = _text(event_name).strip().lower()
    if not text:
        return "unknown"
    if "game changers" in text:
        return "game_changers"
    if any(token in text for token in ["challengers", "academy", "collegiate"]):
        return "tier2"
    if "ascension" in text:
        return "promotion"
    if any(
        token in text
        for token in [
            "valorant champions",
            "valorant masters",
            "masters ",
            "vct ",
            "champions tour",
            "esports world cup",
        ]
    ):
        return "tier1"
    return "unknown"


def infer_event_region(event_name: str, event_series: str = "") -> str:
    text = f"{_text(event_name)} {_text(event_series)}".lower()
    if (
        "masters" in text
        or "esports world cup" in text
        or ("valorant champions" in text and "champions tour" not in text)
    ):
        return "International"
    if any(token in text for token in ["americas", "north america", "latin america", "brazil"]):
        return "VCT Americas"
    if any(token in text for token in ["emea", "europe", "turkey", "mena"]):
        return "VCT EMEA"
    if any(token in text for token in ["pacific", "korea", "japan", "southeast asia", "oceania"]):
        return "VCT Pacific"
    if "china" in text or "cn " in f"{text} ":
        return "VCT China"
    return ""


def estimated_patch_epoch(value) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    periods = math.floor((timestamp - PATCH_EPOCH_ANCHOR).total_seconds() / 86400 / PATCH_EPOCH_DAYS)
    epoch_start = PATCH_EPOCH_ANCHOR + pd.Timedelta(days=periods * PATCH_EPOCH_DAYS)
    return f"estimated-{epoch_start:%Y%m%d}"


def competition_strength(tier: str, region: str) -> float:
    normalized = _text(tier).lower()
    if normalized == "promotion":
        return 0.55
    if normalized != "tier1":
        return 0.35
    return 1.05 if _text(region) == "International" else 1.0


def _region_lookups(registry: pd.DataFrame) -> tuple[dict, dict]:
    by_year = {}
    latest = {}
    if registry.empty:
        return by_year, latest
    for _, row in registry.iterrows():
        name = identity_key(row.get("team"))
        region = str(row.get("region") or row.get("league") or "").strip()
        year = pd.to_numeric(pd.Series([row.get("season_year")]), errors="coerce").iloc[0]
        if not name or not region:
            continue
        latest[name] = region
        if pd.notna(year):
            by_year[(int(year), name)] = region
    return by_year, latest


def _roster_player_lookup(rosters: pd.DataFrame) -> dict[str, int]:
    if rosters.empty or not {"player", "player_id"}.issubset(rosters.columns):
        return {}
    scoped = rosters[["player", "player_id"]].copy()
    scoped["player_key"] = scoped["player"].map(identity_key)
    scoped["player_id"] = pd.to_numeric(scoped["player_id"], errors="coerce")
    scoped = scoped[(scoped["player_key"] != "") & scoped["player_id"].notna()]
    lookup = {}
    for player_key, group in scoped.groupby("player_key"):
        ids = sorted({int(value) for value in group["player_id"]})
        if len(ids) == 1:
            lookup[player_key] = ids[0]
    return lookup


def enrich_match_metadata(
    matches: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    output = matches.copy()
    for column, default in {
        "event_name": "",
        "event_series": "",
        "event_tier": "unknown",
        "competition_tier": "unknown",
        "competition_strength_weight": 1.0,
        "event_region": "",
        "team_region": "",
        "opponent_region": "",
        "patch": "",
        "patch_source": "",
        "player_id": pd.NA,
    }.items():
        if column not in output.columns:
            output[column] = default
    for column in [
        "event_name",
        "event_series",
        "event_tier",
        "competition_tier",
        "event_region",
        "team_region",
        "opponent_region",
        "patch",
        "patch_source",
    ]:
        output[column] = output[column].astype("string").fillna("")

    dates = pd.to_datetime(
        output.get("match_date", pd.Series(index=output.index, dtype=object)),
        utc=True,
        errors="coerce",
    )
    years = pd.to_numeric(
        output.get("season_year", pd.Series(index=output.index, dtype=float)),
        errors="coerce",
    ).fillna(dates.dt.year)
    output["season_year"] = years

    inferred_tiers = output["event_name"].map(classify_event_tier)
    for column in ["event_tier", "competition_tier"]:
        values = output[column].fillna("").astype(str).str.strip().str.lower()
        missing = values.isin({"", "unknown", "nan"})
        output.loc[missing, column] = inferred_tiers[missing]

    inferred_regions = pd.Series(
        [
            infer_event_region(name, series)
            for name, series in zip(output["event_name"], output["event_series"])
        ],
        index=output.index,
    )
    event_regions = output["event_region"].fillna("").astype(str).str.strip()
    output.loc[event_regions.eq(""), "event_region"] = inferred_regions[event_regions.eq("")]

    patches = output["patch"].fillna("").astype(str).str.strip()
    derived_patches = dates.map(estimated_patch_epoch)
    missing_patch = patches.eq("")
    output.loc[missing_patch, "patch"] = derived_patches[missing_patch]
    patch_sources = output["patch_source"].fillna("").astype(str).str.strip()
    output.loc[missing_patch & patch_sources.eq(""), "patch_source"] = "date_epoch"
    output.loc[~missing_patch & patch_sources.eq(""), "patch_source"] = "source"

    registry = registry if registry is not None else pd.DataFrame()
    by_year, latest = _region_lookups(registry)
    for side, name_column in [("team_region", "team"), ("opponent_region", "opponent")]:
        current = output[side].fillna("").astype(str).str.strip()
        missing = current.eq("")
        inferred = []
        for index, name in output[name_column].items():
            year = years.loc[index]
            key = identity_key(name)
            region = by_year.get((int(year), key), "") if pd.notna(year) else ""
            inferred.append(region or latest.get(key, ""))
        inferred_series = pd.Series(inferred, index=output.index)
        output.loc[missing, side] = inferred_series[missing]

    strengths = pd.Series(
        [competition_strength(tier, region) for tier, region in zip(output["competition_tier"], output["event_region"])],
        index=output.index,
    )
    existing_strength = pd.to_numeric(output["competition_strength_weight"], errors="coerce")
    unknown_strength = existing_strength.isna() | existing_strength.eq(1.0) & output["competition_tier"].astype(str).str.lower().ne("tier1")
    output.loc[unknown_strength, "competition_strength_weight"] = strengths[unknown_strength]

    rosters = rosters if rosters is not None else pd.DataFrame()
    player_lookup = _roster_player_lookup(rosters)
    player_ids = pd.to_numeric(output["player_id"], errors="coerce")
    if player_lookup:
        inferred_ids = output.get("player", pd.Series("", index=output.index)).map(
            lambda value: player_lookup.get(identity_key(value))
        )
        player_ids = player_ids.fillna(inferred_ids)
    output["player_id"] = player_ids
    return output


def data_quality_report(matches: pd.DataFrame) -> dict:
    if matches.empty:
        return {"rows": 0, "matches": 0, "maps": 0, "training_ready_matches": 0}

    frame = matches.copy()
    match_ids = pd.to_numeric(frame.get("match_id"), errors="coerce")
    map_ids = pd.to_numeric(frame.get("map_id"), errors="coerce")
    match_level = frame.loc[~match_ids.duplicated(keep="first")].copy()
    map_keys = pd.DataFrame({"match_id": match_ids, "map_id": map_ids}, index=frame.index)
    map_level = frame.loc[map_ids.notna() & ~map_keys.duplicated()].copy()

    def coverage(values) -> float:
        series = pd.Series(values)
        known = series.notna() & series.astype(str).str.strip().ne("") & series.astype(str).str.lower().ne("unknown")
        return float(known.mean()) if len(series) else 0.0

    tiers = match_level.get("competition_tier", pd.Series("unknown", index=match_level.index))
    ready = tiers.fillna("unknown").astype(str).str.lower().isin(TRAINING_TIERS)
    return {
        "rows": int(len(frame)),
        "matches": int(match_ids.nunique()),
        "maps": int(map_ids.nunique()),
        "training_ready_matches": int(match_level.loc[ready, "match_id"].nunique()) if "match_id" in match_level else 0,
        "player_id_coverage": coverage(frame.get("player_id", pd.Series(index=frame.index, dtype=object))),
        "event_metadata_coverage": min(
            coverage(match_level.get("event_id", pd.Series(index=match_level.index, dtype=object))),
            coverage(match_level.get("event_name", pd.Series(index=match_level.index, dtype=object))),
        ),
        "tier_coverage": coverage(tiers),
        "patch_coverage": coverage(match_level.get("patch", pd.Series(index=match_level.index, dtype=object))),
        "map_pick_coverage": coverage(map_level.get("map_pick_type", pd.Series(index=map_level.index, dtype=object))),
        "map_score_coverage": float(
            (
                pd.to_numeric(
                    map_level.get(
                        "map_team_score",
                        pd.Series(index=map_level.index, dtype=float),
                    ),
                    errors="coerce",
                ).notna()
                & pd.to_numeric(
                    map_level.get(
                        "map_opp_score",
                        pd.Series(index=map_level.index, dtype=float),
                    ),
                    errors="coerce",
                ).notna()
            ).mean()
        ) if len(map_level) else 0.0,
        "duplicate_player_map_rows": int(
            frame.loc[map_ids.notna()].duplicated(
                subset=[column for column in ["match_id", "map_id", "team", "player"] if column in frame.columns]
            ).sum()
        ),
    }


def filter_training_ready_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    output = matches.copy()
    tiers = output.get("competition_tier", pd.Series("unknown", index=output.index)).fillna("unknown").astype(str).str.lower()
    event_tiers = output.get("event_tier", pd.Series("unknown", index=output.index)).fillna("unknown").astype(str).str.lower()
    classified = tiers.isin(TRAINING_TIERS) | event_tiers.isin(TRAINING_TIERS)
    output = output[classified].copy()
    if output.empty:
        return output

    key_column = "match_key" if "match_key" in output.columns else "match_id"
    team_values = output.get("team", pd.Series("", index=output.index)).fillna("").astype(str)
    player_values = output.get("player", pd.Series("", index=output.index)).fillna("").astype(str)
    if "team" not in output.columns:
        output["team"] = team_values
    if "player" not in output.columns:
        output["player"] = player_values
    output["_player_identity"] = team_values + "\x00" + player_values
    date_source = output.get(
        "match_date_sort",
        output.get("match_date", pd.Series(index=output.index, dtype=object)),
    )
    output["_date_known"] = pd.to_datetime(
        date_source,
        utc=True,
        errors="coerce",
    ).notna()
    output["_team_score_known"] = pd.to_numeric(
        output.get("team_score", pd.Series(index=output.index, dtype=float)),
        errors="coerce",
    ).notna()
    output["_opponent_score_known"] = pd.to_numeric(
        output.get("opp_score", pd.Series(index=output.index, dtype=float)),
        errors="coerce",
    ).notna()
    summary = output.groupby(key_column, sort=False).agg(
        teams=("team", "nunique"),
        players=("_player_identity", "nunique"),
        date_known=("_date_known", "max"),
        team_score_known=("_team_score_known", "max"),
        opponent_score_known=("_opponent_score_known", "max"),
    )
    bad_teams = summary["teams"].ne(2)
    bad_players = ~bad_teams & summary["players"].lt(10)
    bad_dates = ~bad_teams & ~bad_players & ~summary["date_known"]
    bad_scores = (
        ~bad_teams
        & ~bad_players
        & ~bad_dates
        & (~summary["team_score_known"] | ~summary["opponent_score_known"])
    )
    valid_keys = summary.index[
        ~(bad_teams | bad_players | bad_dates | bad_scores)
    ].tolist()
    rejection_counts = {
        "teams": int(bad_teams.sum()),
        "players": int(bad_players.sum()),
        "date": int(bad_dates.sum()),
        "score": int(bad_scores.sum()),
    }

    filtered = output[output[key_column].isin(valid_keys)].copy()
    filtered = filtered.drop(
        columns=[
            "_player_identity",
            "_date_known",
            "_team_score_known",
            "_opponent_score_known",
        ],
        errors="ignore",
    )
    filtered.attrs.update(matches.attrs)
    filtered.attrs["training_validation"] = {
        "input_matches": int(matches[key_column].nunique()),
        "classified_matches": int(output[key_column].nunique()),
        "training_ready_matches": int(len(valid_keys)),
        "rejected": rejection_counts,
    }
    return filtered.reset_index(drop=True)
