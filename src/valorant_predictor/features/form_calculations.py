import math
import re

import pandas as pd


DEFAULT_DECAY = 0.85
PRIOR_MAPS = 8
PLAYER_FORM_DAYS = 60
PLAYER_FORM_HALF_LIFE_DAYS = 30.0
PLAYER_STABLE_HALF_LIFE_DAYS = 180.0
PLAYER_FRESHNESS_DAYS = 45.0
PLAYER_VOLATILITY_PRIOR = 0.18
CURATED_COMPETITION_TIERS = {"tier1", "promotion"}


def normalize_player_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalize_map_name(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return re.sub(
        r"\s*[\[(]?(?:pick(?:ed)?|decider)[\])]?$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def match_id_from_url(url: str) -> int | None:
    match = re.search(r"vlr\.gg/(\d+)|/(\d+)/", str(url))
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def parse_match_datetime(value) -> pd.Timestamp:
    if pd.isna(value) or value == "":
        return pd.NaT

    text = str(value).strip()
    if re.fullmatch(r"\d{10}", text):
        return pd.to_datetime(int(text), unit="s", utc=True, errors="coerce")
    if re.fullmatch(r"\d{13}", text):
        return pd.to_datetime(int(text), unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(text, utc=True, errors="coerce")


def filter_matches_by_time(
    matches: pd.DataFrame,
    season_year: int | None = None,
    recent_days: int | None = None,
    reference_date: str | None = None,
) -> pd.DataFrame:
    if matches.empty:
        return matches

    output = clean_match_data(matches)
    if output["match_date_sort"].isna().all():
        return output

    if season_year is not None:
        output = output[output["match_date_sort"].dt.year == season_year].copy()

    if recent_days is not None:
        if reference_date:
            reference = parse_match_datetime(reference_date)
        else:
            reference = output["match_date_sort"].max()
        if pd.notna(reference):
            cutoff = reference - pd.Timedelta(days=recent_days)
            output = output[output["match_date_sort"] >= cutoff].copy()

    return output.reset_index(drop=True)


def filter_curated_competition_history(
    matches: pd.DataFrame,
    minimum_classified_matches: int = 20,
) -> pd.DataFrame:
    if matches.empty or "competition_tier" not in matches.columns:
        return matches
    output = matches.copy()
    dates = pd.to_datetime(output.get("match_date"), utc=True, errors="coerce")
    tiers = output["competition_tier"].fillna("").astype(str).str.lower()
    event_tiers = output.get(
        "event_tier",
        pd.Series("", index=output.index),
    ).fillna("").astype(str).str.lower()
    allowed = tiers.isin(CURATED_COMPETITION_TIERS) | event_tiers.eq("tier1")
    match_ids = pd.to_numeric(output.get("match_id"), errors="coerce")
    restricted_years = []
    for year in sorted(dates.dt.year.dropna().astype(int).unique()):
        count = match_ids[dates.dt.year.eq(year) & allowed].nunique()
        if count >= minimum_classified_matches:
            restricted_years.append(year)
    if not restricted_years:
        return output
    latest_curated_year = max(restricted_years)
    keep = dates.dt.year.gt(latest_curated_year) | allowed
    filtered = output[keep].copy()
    filtered.attrs["curated_competition_years"] = restricted_years
    filtered.attrs["excluded_uncurated_rows"] = int((~keep).sum())
    return filtered.reset_index(drop=True)


def fallback_rating_from_stats(df: pd.DataFrame) -> pd.Series:
    deaths = df["deaths"].replace(0, 1)
    kd_ratio = df["kills"] / deaths
    acs_component = (df["acs"] - 200.0) / 250.0
    kd_component = kd_ratio - 1.0
    assist_component = (df["assists"] - 5.0) / 35.0

    return 1.00 + (0.34 * acs_component) + (0.28 * kd_component) + (0.08 * assist_component)


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    required = ["team", "opponent", "player", "acs", "kills", "deaths"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Match CSV is missing columns: {', '.join(missing)}")

    prepared_columns = {
        "match_key",
        "match_date_sort",
        "season_year",
        "rating_for_model",
        "match_importance",
        "competition_strength_weight",
        "patch",
    }
    if prepared_columns.issubset(df.columns) and pd.api.types.is_datetime64_any_dtype(
        df["match_date_sort"]
    ):
        return df.copy()

    output = df.copy()
    if "map_name" in output.columns:
        output["map_name"] = output["map_name"].apply(normalize_map_name)
    if "player_id" not in output.columns:
        output["player_id"] = pd.NA
    output["player_id"] = pd.to_numeric(output["player_id"], errors="coerce")
    for column, default in {
        "event_name": "",
        "event_series": "",
        "event_stage": "",
        "event_tier": "unknown",
        "is_lan": False,
        "patch": "",
        "map_veto": "",
        "map_pick_team": "",
        "map_pick_type": "unknown",
        "map_veto_order": pd.NA,
        "map_data_source": "legacy",
        "match_importance": 1.0,
        "competition_tier": "unknown",
        "competition_strength_weight": 1.0,
        "team_tier_at_match": "unknown",
        "opponent_tier_at_match": "unknown",
        "team_promoted_next_season": False,
        "opponent_promoted_next_season": False,
        "data_source": "legacy",
    }.items():
        if column not in output.columns:
            output[column] = default
    for col in ["acs", "kills", "deaths", "assists", "vlr_rating", "team_score", "opp_score"]:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")
    output["match_importance"] = pd.to_numeric(
        output["match_importance"], errors="coerce"
    ).fillna(1.0).clip(0.75, 1.15)
    output["competition_strength_weight"] = pd.to_numeric(
        output["competition_strength_weight"], errors="coerce"
    ).fillna(1.0).clip(0.25, 1.25)

    if "assists" not in output.columns:
        output["assists"] = 0
    if "vlr_rating" not in output.columns:
        output["vlr_rating"] = pd.NA
    if "is_winner" not in output.columns:
        output["is_winner"] = output.get("team", "") == output.get("winner", "")
    else:
        output["is_winner"] = output["is_winner"].astype(str).str.lower().isin(["true", "1", "yes"])

    if "match_id" not in output.columns:
        output["match_id"] = output["match_url"].apply(match_id_from_url)
    else:
        output["match_id"] = pd.to_numeric(output["match_id"], errors="coerce")
        missing_match_ids = output["match_id"].isna()
        if missing_match_ids.any():
            output.loc[missing_match_ids, "match_id"] = output.loc[
                missing_match_ids, "match_url"
            ].apply(match_id_from_url)

    if "map_id" not in output.columns:
        output["map_id"] = pd.NA
    output["map_id"] = pd.to_numeric(output["map_id"], errors="coerce")
    if "map_number" not in output.columns:
        output["map_number"] = pd.NA
    output["map_number"] = pd.to_numeric(output["map_number"], errors="coerce")
    output["map_veto_order"] = pd.to_numeric(output["map_veto_order"], errors="coerce")

    if "stat_scope" not in output.columns:
        output["stat_scope"] = output["map_id"].notna().map({True: "map", False: "legacy"})
    else:
        scope = output["stat_scope"].fillna("").astype(str).str.strip().str.lower()
        output["stat_scope"] = scope.where(
            scope != "",
            output["map_id"].notna().map({True: "map", False: "legacy"}),
        )

    output = output[output["stat_scope"] != "aggregate"].copy()
    mapped_match_ids = set(output.loc[output["stat_scope"] == "map", "match_id"].dropna())
    if mapped_match_ids:
        output = output[
            ~(
                output["match_id"].isin(mapped_match_ids)
                & (output["stat_scope"] != "map")
            )
        ].copy()

    map_rows = output[output["map_id"].notna()].drop_duplicates(
        subset=["match_id", "map_id", "team", "player"],
        keep="last",
    )
    legacy_subset = [
        "match_id",
        "team",
        "player",
        "vlr_rating",
        "acs",
        "kills",
        "deaths",
        "assists",
    ]
    if "agents" in output.columns:
        legacy_subset.insert(3, "agents")
    legacy_rows = output[output["map_id"].isna()].drop_duplicates(
        subset=legacy_subset,
        keep="last",
    )
    output = pd.concat([legacy_rows, map_rows], ignore_index=True)
    output["match_key"] = ""
    known_match_ids = output["match_id"].notna()
    output.loc[known_match_ids, "match_key"] = (
        "id:" + output.loc[known_match_ids, "match_id"].astype("int64").astype(str)
    )
    missing_keys = output["match_key"] == ""
    output.loc[missing_keys, "match_key"] = "url:" + output.loc[missing_keys, "match_url"].astype(str).str.split("?").str[0].str.rstrip("/")

    if "match_date" in output.columns:
        date_text = output["match_date"].astype("string").str.strip()
        output["match_date_sort"] = pd.to_datetime(date_text, utc=True, errors="coerce")
        seconds = date_text.str.fullmatch(r"\d{10}", na=False)
        milliseconds = date_text.str.fullmatch(r"\d{13}", na=False)
        if seconds.any():
            output.loc[seconds, "match_date_sort"] = pd.to_datetime(
                pd.to_numeric(date_text[seconds], errors="coerce"),
                unit="s",
                utc=True,
                errors="coerce",
            )
        if milliseconds.any():
            output.loc[milliseconds, "match_date_sort"] = pd.to_datetime(
                pd.to_numeric(date_text[milliseconds], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            )
    else:
        output["match_date_sort"] = pd.NaT

    output["season_year"] = output["match_date_sort"].dt.year

    output["rating_for_model"] = output["vlr_rating"].where(
        output["vlr_rating"].notna(),
        fallback_rating_from_stats(output),
    )
    output["rating_for_model"] = output["rating_for_model"].clip(lower=0.35, upper=1.90)

    output = output.dropna(subset=["team", "player", "acs", "kills", "deaths", "rating_for_model"])
    return output.sort_values(
        ["match_date_sort", "match_id", "map_number"],
        na_position="first",
    ).reset_index(drop=True)


def weighted_recent_mean(values: pd.Series, decay: float = DEFAULT_DECAY) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().tolist()
    if not clean:
        return 0.0

    weights = [decay ** (len(clean) - index - 1) for index in range(len(clean))]
    return sum(value * weight for value, weight in zip(clean, weights)) / sum(weights)


def shrink_to_prior(value: float, sample_size: int, prior: float, prior_maps: int = PRIOR_MAPS) -> tuple[float, float]:
    reliability = sample_size / (sample_size + prior_maps) if sample_size > 0 else 0.0
    return prior + reliability * (value - prior), reliability


def time_decay_weights(
    dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_days: float,
) -> pd.Series:
    parsed = dates.apply(parse_match_datetime)
    ages = (reference_date - parsed).dt.total_seconds().div(86400).clip(lower=0)
    weights = (0.5 ** (ages / half_life_days)).where(parsed.notna(), 0.35)
    return weights.fillna(0.35).astype(float)


def weighted_mean_with_weights(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return 0.0
    return float((numeric[valid] * weights[valid]).sum() / weights[valid].sum())


def effective_sample_size(weights: pd.Series) -> float:
    clean = pd.to_numeric(weights, errors="coerce").dropna()
    clean = clean[clean > 0]
    if clean.empty:
        return 0.0
    return float(clean.sum() ** 2 / clean.pow(2).sum())


def player_form_from_group(
    group: pd.DataFrame,
    global_rating: float,
    recent_maps: int,
    reference_date: pd.Timestamp | None = None,
) -> dict:
    ordered = group.sort_values(["match_date_sort", "match_id", "map_number"], na_position="first")
    latest_date = ordered["match_date_sort"].dropna().max() if ordered["match_date_sort"].notna().any() else pd.NaT
    if reference_date is None:
        reference_date = pd.Timestamp.now(tz="UTC")
    else:
        reference_date = parse_match_datetime(reference_date)
    if pd.notna(latest_date) and latest_date > reference_date:
        reference_date = latest_date

    if pd.notna(latest_date):
        cutoff = reference_date - pd.Timedelta(days=PLAYER_FORM_DAYS)
        recent_window = ordered[ordered["match_date_sort"] >= cutoff].copy()
    else:
        recent_window = ordered.tail(max(recent_maps, 10)).copy()
    if recent_window.empty:
        recent_window = ordered.tail(max(recent_maps, 10)).copy()

    recent = recent_window.tail(recent_maps)
    last_3 = ordered.tail(3)
    last_5 = ordered.tail(5)
    last_10 = ordered.tail(10)

    recent_weights = time_decay_weights(
        recent_window["match_date_sort"],
        reference_date,
        PLAYER_FORM_HALF_LIFE_DAYS,
    )
    stable_weights = time_decay_weights(
        ordered["match_date_sort"],
        reference_date,
        PLAYER_STABLE_HALF_LIFE_DAYS,
    )
    form_weights = recent_weights.loc[recent.index]
    weighted_form = weighted_mean_with_weights(recent["rating_for_model"], form_weights)
    recent_window_rating = weighted_mean_with_weights(recent_window["rating_for_model"], recent_weights)
    stable_rating = weighted_mean_with_weights(ordered["rating_for_model"], stable_weights)
    last_3_rating = weighted_recent_mean(last_3["rating_for_model"])
    last_5_rating = weighted_recent_mean(last_5["rating_for_model"])
    last_10_rating = weighted_recent_mean(last_10["rating_for_model"])
    overall_rating = ordered["rating_for_model"].mean()

    if last_10_rating:
        blended = (0.45 * last_5_rating) + (0.35 * recent_window_rating) + (0.20 * stable_rating)
    else:
        blended = weighted_form

    effective_maps = effective_sample_size(recent_weights)
    sample_reliability = effective_maps / (effective_maps + PRIOR_MAPS) if effective_maps > 0 else 0.0
    days_since_last_match = (
        max(0.0, float((reference_date - latest_date).total_seconds() / 86400))
        if pd.notna(latest_date)
        else float("inf")
    )
    freshness = math.exp(-days_since_last_match / PLAYER_FRESHNESS_DAYS) if math.isfinite(days_since_last_match) else 0.0
    data_reliability = max(0.0, min(1.0, sample_reliability * freshness))
    base_rating = global_rating + data_reliability * (blended - global_rating)
    trend = last_3_rating - last_10_rating if last_10_rating else 0.0

    deaths = recent["deaths"].replace(0, 1)
    last_seen_match_id = ordered["match_id"].dropna().iloc[-1] if ordered["match_id"].notna().any() else None
    volatility_sample = ordered.tail(20)["rating_for_model"]
    observed_std = float(volatility_sample.std(ddof=0)) if len(volatility_sample) > 1 else PLAYER_VOLATILITY_PRIOR
    volatility_weight = len(volatility_sample) / (len(volatility_sample) + PRIOR_MAPS)
    rating_std = math.sqrt(
        volatility_weight * observed_std**2
        + (1.0 - volatility_weight) * PLAYER_VOLATILITY_PRIOR**2
    )
    agents = []
    if "agents" in ordered.columns:
        for value in ordered.tail(20)["agents"].dropna().astype(str):
            agents.extend(agent.strip() for agent in value.split(";") if agent.strip())
    agent_counts = pd.Series(agents, dtype="object").value_counts() if agents else pd.Series(dtype="int64")

    return {
        "base_rating": max(0.45, min(1.70, base_rating)),
        "raw_form_rating": weighted_form,
        "recent_60d_rating": recent_window_rating,
        "stable_rating": stable_rating,
        "last_3_rating": last_3_rating,
        "last_5_rating": last_5_rating,
        "last_10_rating": last_10_rating,
        "overall_rating": overall_rating,
        "rating_trend": trend,
        "avg_vlr_rating": recent["vlr_rating"].mean() if recent["vlr_rating"].notna().any() else None,
        "avg_acs": recent["acs"].mean(),
        "kd_ratio": (recent["kills"] / deaths).mean(),
        "avg_assists": recent["assists"].mean(),
        "win_rate": recent["is_winner"].astype(float).mean(),
        "recent_maps": len(recent),
        "total_maps": len(ordered),
        "maps_60d": len(recent_window),
        "effective_maps": effective_maps,
        "days_since_last_match": days_since_last_match,
        "freshness": freshness,
        "data_reliability": data_reliability,
        "reliability": data_reliability,
        "rating_std": rating_std,
        "agent_pool_size": int(len(agent_counts)),
        "primary_agents": ";".join(agent_counts.head(3).index.astype(str)),
        "lineup_certainty": 1.0,
        "last_seen_match_id": last_seen_match_id,
    }


def build_player_profiles(matches: pd.DataFrame, teams: list[str], recent_maps: int = 10) -> pd.DataFrame:
    cleaned = clean_match_data(matches)
    scoped = cleaned[cleaned["team"].isin(teams)].copy()
    if scoped.empty:
        return pd.DataFrame()

    global_rating = cleaned["rating_for_model"].mean()
    rows = []
    for (team, player), group in scoped.groupby(["team", "player"], sort=True):
        stats = player_form_from_group(group, global_rating, recent_maps)
        rows.append({"team": team, "player": player, **stats})
    return pd.DataFrame(rows)


def clean_roster_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    output = df.copy()
    if "is_active_player" not in output.columns:
        output["is_active_player"] = True
    else:
        output["is_active_player"] = output["is_active_player"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    if "is_staff" not in output.columns:
        output["is_staff"] = False
    else:
        output["is_staff"] = output["is_staff"].astype(str).str.lower().isin(["true", "1", "yes"])

    if "status" not in output.columns:
        output["status"] = "active"
    if "roster_role" not in output.columns:
        output["roster_role"] = output["status"]

    output["player_norm"] = output["player"].apply(normalize_player_name)
    return output.dropna(subset=["team", "player"])


def default_player_profile(global_rating: float) -> dict:
    return {
        "base_rating": global_rating,
        "raw_form_rating": global_rating,
        "recent_60d_rating": global_rating,
        "stable_rating": global_rating,
        "last_3_rating": global_rating,
        "last_5_rating": global_rating,
        "last_10_rating": global_rating,
        "overall_rating": global_rating,
        "rating_trend": 0.0,
        "avg_vlr_rating": None,
        "avg_acs": 200.0,
        "kd_ratio": 1.0,
        "avg_assists": 5.0,
        "win_rate": 0.5,
        "recent_maps": 0,
        "total_maps": 0,
        "maps_60d": 0,
        "effective_maps": 0.0,
        "days_since_last_match": float("inf"),
        "freshness": 0.0,
        "data_reliability": 0.0,
        "reliability": 0.0,
        "rating_std": PLAYER_VOLATILITY_PRIOR,
        "agent_pool_size": 0,
        "primary_agents": "",
        "lineup_certainty": 0.0,
        "last_seen_match_id": None,
    }


def build_player_profiles_from_rosters(
    matches: pd.DataFrame,
    teams: list[str],
    roster_df: pd.DataFrame,
    recent_maps: int = 10,
    active_only: bool = True,
) -> pd.DataFrame:
    roster = clean_roster_data(roster_df)
    if roster.empty:
        return build_player_profiles(matches, teams, recent_maps=recent_maps)

    roster = roster[roster["team"].isin(teams) & ~roster["is_staff"]].copy()
    if active_only:
        roster = roster[roster["is_active_player"]].copy()
    if roster.empty:
        return build_player_profiles(matches, teams, recent_maps=recent_maps)

    cleaned = clean_match_data(matches)
    if cleaned.empty:
        return pd.DataFrame()

    cleaned["player_norm"] = cleaned["player"].apply(normalize_player_name)
    cleaned["player_id"] = pd.to_numeric(cleaned.get("player_id"), errors="coerce")
    global_rating = cleaned["rating_for_model"].mean()

    rows = []
    for _, roster_row in roster.iterrows():
        player_norm = roster_row["player_norm"]
        roster_player_id = pd.to_numeric(
            pd.Series([roster_row.get("player_id")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(roster_player_id) and cleaned["player_id"].notna().any():
            player_matches = cleaned[cleaned["player_id"] == roster_player_id].copy()
            if player_matches.empty:
                player_matches = cleaned[cleaned["player_norm"] == player_norm].copy()
        else:
            player_matches = cleaned[cleaned["player_norm"] == player_norm].copy()
        current_team_matches = player_matches[player_matches["team"] == roster_row["team"]]

        if player_matches.empty:
            stats = default_player_profile(global_rating)
            historical_teams = ""
            roster_uncertainty = 0.22
            is_new_to_team = True
            no_player_history = True
            team_continuity = 0.0
        else:
            stats = player_form_from_group(player_matches, global_rating, recent_maps)
            historical_teams = ";".join(sorted(player_matches["team"].dropna().unique()))
            is_new_to_team = current_team_matches.empty
            no_player_history = False
            current_team_maps = int(current_team_matches["map_id"].nunique()) if "map_id" in current_team_matches else len(current_team_matches)
            team_continuity = current_team_maps / (current_team_maps + PRIOR_MAPS) if current_team_maps > 0 else 0.0
            roster_uncertainty = 0.12 * (1.0 - team_continuity)

        lineup_certainty = max(0.0, min(1.0, 1.0 - roster_uncertainty))
        stats["lineup_certainty"] = lineup_certainty
        stats["reliability"] = stats.get("data_reliability", 0.0) * lineup_certainty

        rows.append(
            {
                "team": roster_row["team"],
                "player": roster_row["player"],
                "player_id": roster_row.get("player_id"),
                "roster_status": roster_row.get("status", "active"),
                "roster_role": roster_row.get("roster_role", "active"),
                "is_current_roster": True,
                "is_new_to_team": is_new_to_team,
                "no_player_history": no_player_history,
                "team_history_maps": len(current_team_matches),
                "team_continuity": team_continuity,
                "historical_teams": historical_teams,
                "roster_uncertainty": roster_uncertainty,
                **stats,
            }
        )

    return pd.DataFrame(rows)


def team_map_rows(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    cleaned = clean_match_data(matches)
    scoped = cleaned[cleaned["team"] == team].copy()
    if scoped.empty:
        return pd.DataFrame()

    rows = []
    for (match_key, row_team), group in scoped.groupby(["match_key", "team"], sort=False):
        team_score = group["team_score"].dropna().iloc[0] if "team_score" in group and group["team_score"].notna().any() else None
        opp_score = group["opp_score"].dropna().iloc[0] if "opp_score" in group and group["opp_score"].notna().any() else None
        score_margin = 0.0
        if team_score is not None and opp_score is not None:
            score_margin = float(team_score - opp_score)

        rows.append(
            {
                "team": row_team,
                "match_key": match_key,
                "match_url": group["match_url"].dropna().iloc[0],
                "match_id": group["match_id"].dropna().max(),
                "match_date_sort": group["match_date_sort"].dropna().max() if group["match_date_sort"].notna().any() else pd.NaT,
                "team_avg_rating": group["rating_for_model"].mean(),
                "team_win": group["is_winner"].astype(float).max(),
                "score_margin": score_margin,
            }
        )

    return pd.DataFrame(rows).sort_values(["match_date_sort", "match_id"], na_position="first")


def calculate_team_form(matches: pd.DataFrame, team: str, recent_matches: int = 10) -> dict:
    rows = team_map_rows(matches, team)
    if rows.empty:
        return {
            "recent_win_rate": 0.5,
            "team_recent_rating": 1.0,
            "team_rating_trend": 0.0,
            "score_margin": 0.0,
            "consistency_penalty": 0.0,
            "team_form_adjustment": 0.0,
            "recent_team_maps": 0,
        }

    recent = rows.tail(recent_matches)
    last_3 = rows.tail(3)
    recent_win_rate = weighted_recent_mean(recent["team_win"])
    team_recent_rating = weighted_recent_mean(recent["team_avg_rating"])
    score_margin = weighted_recent_mean(recent["score_margin"]) / 3.0
    trend = weighted_recent_mean(last_3["team_avg_rating"]) - team_recent_rating
    consistency = float(recent["team_avg_rating"].std()) if len(recent) > 1 else 0.0

    adjustment = 0.08 * (recent_win_rate - 0.5)
    adjustment += 0.05 * score_margin
    adjustment += 0.06 * trend
    adjustment -= 0.03 * min(consistency, 0.25)

    return {
        "recent_win_rate": recent_win_rate,
        "team_recent_rating": team_recent_rating,
        "team_rating_trend": trend,
        "score_margin": score_margin,
        "consistency_penalty": consistency,
        "team_form_adjustment": adjustment,
        "recent_team_maps": len(recent),
    }


def probable_lineup(player_predictions: pd.DataFrame, team: str, lineup_size: int = 5) -> pd.DataFrame:
    scoped = player_predictions[player_predictions["team"] == team].copy()
    if scoped.empty:
        return scoped

    sort_cols = ["last_seen_match_id", "recent_maps", "predicted_rating"]
    for col in sort_cols:
        if col not in scoped.columns:
            scoped[col] = 0
    return scoped.sort_values(sort_cols, ascending=[False, False, False]).head(lineup_size)


def logistic_probability(diff: float, scale: float = 6.5) -> float:
    return 1.0 / (1.0 + math.exp(-diff * scale))


def best_of_probability(single_map_probability: float, best_of: int = 3) -> float:
    if best_of <= 1:
        return single_map_probability
    if best_of % 2 == 0:
        raise ValueError("best_of must be odd, such as 1, 3, or 5.")

    needed = (best_of // 2) + 1
    probability = 0.0
    for wins in range(needed, best_of + 1):
        probability += math.comb(best_of, wins) * (single_map_probability ** wins) * (
            (1.0 - single_map_probability) ** (best_of - wins)
        )
    return probability
