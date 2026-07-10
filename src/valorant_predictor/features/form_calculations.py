import math
import re

import pandas as pd


DEFAULT_DECAY = 0.85
PRIOR_MAPS = 8


def normalize_player_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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

    output = df.copy()
    for col in ["acs", "kills", "deaths", "assists", "vlr_rating", "team_score", "opp_score"]:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")

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

    if "match_date" in output.columns:
        output["match_date_sort"] = output["match_date"].apply(parse_match_datetime)
    else:
        output["match_date_sort"] = pd.NaT

    output["season_year"] = output["match_date_sort"].dt.year

    output["rating_for_model"] = output["vlr_rating"].where(
        output["vlr_rating"].notna(),
        fallback_rating_from_stats(output),
    )
    output["rating_for_model"] = output["rating_for_model"].clip(lower=0.35, upper=1.90)

    output = output.dropna(subset=["team", "player", "acs", "kills", "deaths", "rating_for_model"])
    return output.sort_values(["match_date_sort", "match_id"], na_position="first").reset_index(drop=True)


def weighted_recent_mean(values: pd.Series, decay: float = DEFAULT_DECAY) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().tolist()
    if not clean:
        return 0.0

    weights = [decay ** (len(clean) - index - 1) for index in range(len(clean))]
    return sum(value * weight for value, weight in zip(clean, weights)) / sum(weights)


def shrink_to_prior(value: float, sample_size: int, prior: float, prior_maps: int = PRIOR_MAPS) -> tuple[float, float]:
    reliability = sample_size / (sample_size + prior_maps) if sample_size > 0 else 0.0
    return prior + reliability * (value - prior), reliability


def player_form_from_group(group: pd.DataFrame, global_rating: float, recent_maps: int) -> dict:
    ordered = group.sort_values(["match_date_sort", "match_id"], na_position="first")
    recent = ordered.tail(recent_maps)
    last_3 = ordered.tail(3)
    last_5 = ordered.tail(5)
    last_10 = ordered.tail(10)

    weighted_form = weighted_recent_mean(recent["rating_for_model"])
    last_3_rating = weighted_recent_mean(last_3["rating_for_model"])
    last_5_rating = weighted_recent_mean(last_5["rating_for_model"])
    last_10_rating = weighted_recent_mean(last_10["rating_for_model"])
    overall_rating = ordered["rating_for_model"].mean()

    if last_10_rating:
        blended = (0.50 * last_5_rating) + (0.30 * last_10_rating) + (0.20 * overall_rating)
    else:
        blended = weighted_form

    base_rating, reliability = shrink_to_prior(blended, len(recent), global_rating)
    trend = last_3_rating - last_10_rating if last_10_rating else 0.0

    deaths = recent["deaths"].replace(0, 1)
    last_seen_match_id = ordered["match_id"].dropna().iloc[-1] if ordered["match_id"].notna().any() else None

    return {
        "base_rating": max(0.45, min(1.70, base_rating)),
        "raw_form_rating": weighted_form,
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
        "reliability": reliability,
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
        "reliability": 0.0,
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
    global_rating = cleaned["rating_for_model"].mean()

    rows = []
    for _, roster_row in roster.iterrows():
        player_norm = roster_row["player_norm"]
        player_matches = cleaned[cleaned["player_norm"] == player_norm].copy()
        current_team_matches = player_matches[player_matches["team"] == roster_row["team"]]

        if player_matches.empty:
            stats = default_player_profile(global_rating)
            historical_teams = ""
            roster_uncertainty = 0.22
            is_new_to_team = True
            no_player_history = True
        else:
            stats = player_form_from_group(player_matches, global_rating, recent_maps)
            historical_teams = ";".join(sorted(player_matches["team"].dropna().unique()))
            is_new_to_team = current_team_matches.empty
            no_player_history = False
            roster_uncertainty = 0.12 if is_new_to_team else 0.0

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
    for (match_url, row_team), group in scoped.groupby(["match_url", "team"], sort=False):
        team_score = group["team_score"].dropna().iloc[0] if "team_score" in group and group["team_score"].notna().any() else None
        opp_score = group["opp_score"].dropna().iloc[0] if "opp_score" in group and group["opp_score"].notna().any() else None
        score_margin = 0.0
        if team_score is not None and opp_score is not None:
            score_margin = float(team_score - opp_score)

        rows.append(
            {
                "team": row_team,
                "match_url": match_url,
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
