import math
from copy import deepcopy
from collections import defaultdict

import pandas as pd

from .form_calculations import normalize_player_name


INITIAL_ELO = 1500.0
ELO_SCALE = 400.0
ELO_K = 28.0
REGION_ELO_K = 16.0
REGION_ELO_WEIGHT = 0.35
MAP_ELO_K = 22.0
TEAM_ELO_HALF_LIFE_DAYS = 120.0
MAP_POOL_HALF_LIFE_DAYS = 90.0
PLAYER_SKILL_HALF_LIFE_DAYS = 120.0
SEASON_CARRY = 0.85
PATCH_CARRY = 0.92
MIN_ROSTER_CARRY = 0.55
MAP_PRIOR_GAMES = 4.0
MAP_PRIOR_RATING = 1.0
MAP_HISTORY_PRIOR_GAMES = 10.0
MAX_MAP_SPECIFIC_WEIGHT = 0.35

SEQUENTIAL_TEAM_FEATURES = [
    "elo_diff",
    "team_elo_diff",
    "elo_matches",
    "elo_reliability",
    "minimum_elo_freshness",
    "strength_of_schedule_diff",
    "region_elo_diff",
    "region_reliability",
    "cross_region_match",
    "map_pool_win_rate_diff",
    "map_pool_rating_diff",
    "map_pool_elo_diff",
    "map_pool_maps",
    "map_pool_reliability",
    "map_selection_reliability",
    "roster_continuity_diff",
    "minimum_roster_continuity",
    "lineup_rating_diff",
    "lineup_reliability",
    "same_patch_context",
]


def elo_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def _logit(probability: float) -> float:
    probability = min(0.999, max(0.001, probability))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def map_history_reliability(
    team_maps: float,
    opponent_maps: float,
    prior_games: float = MAP_HISTORY_PRIOR_GAMES,
) -> float:
    team_maps = max(0.0, float(team_maps))
    opponent_maps = max(0.0, float(opponent_maps))
    prior_games = max(0.01, float(prior_games))
    team_reliability = team_maps / (team_maps + prior_games)
    opponent_reliability = opponent_maps / (opponent_maps + prior_games)
    return math.sqrt(team_reliability * opponent_reliability)


def hierarchical_map_probability(
    team_probability: float,
    map_signal_probability: float,
    reliability: float,
    max_map_weight: float = MAX_MAP_SPECIFIC_WEIGHT,
) -> float:
    reliability = min(1.0, max(0.0, float(reliability)))
    map_weight = min(1.0, max(0.0, float(max_map_weight))) * reliability
    combined_logit = (
        (1.0 - map_weight) * _logit(float(team_probability))
        + map_weight * _logit(float(map_signal_probability))
    )
    return min(0.95, max(0.05, _sigmoid(combined_logit)))


def rebase_map_probability(
    map_probability: float,
    previous_team_probability: float,
    current_team_probability: float,
) -> float:
    map_adjustment = _logit(float(map_probability)) - _logit(
        float(previous_team_probability)
    )
    return min(
        0.95,
        max(0.05, _sigmoid(_logit(float(current_team_probability)) + map_adjustment)),
    )


def _new_state() -> dict:
    return {
        "elo": defaultdict(lambda: INITIAL_ELO),
        "opponent_elo_sum": defaultdict(float),
        "opponent_count": defaultdict(int),
        "map_stats": defaultdict(
            lambda: {
                "wins": 0.0,
                "maps": 0.0,
                "rating_sum": 0.0,
                "picks": 0.0,
                "deciders": 0.0,
                "last_date": None,
                "patch": "",
            }
        ),
        "map_elo": defaultdict(lambda: INITIAL_ELO),
        "map_elo_matches": defaultdict(float),
        "map_elo_last_date": {},
        "region_elo": defaultdict(lambda: INITIAL_ELO),
        "region_matches": defaultdict(float),
        "team_regions": {},
        "player_skill": defaultdict(lambda: 1.0),
        "player_maps": defaultdict(float),
        "player_last_date": {},
        "lineups": {},
        "elo_last_date": {},
        "elo_freshness": {},
        "elo_season": {},
        "last_patch": {},
        "sos_last_date": {},
    }


def _upgrade_state(state: dict | None) -> dict:
    upgraded = _new_state()
    if not state:
        return upgraded
    for key, value in state.items():
        if key in upgraded:
            upgraded[key] = deepcopy(value)
    for key, factory in [
        ("elo", lambda: INITIAL_ELO),
        ("opponent_elo_sum", float),
        ("opponent_count", int),
        ("map_elo", lambda: INITIAL_ELO),
        ("map_elo_matches", float),
        ("region_elo", lambda: INITIAL_ELO),
        ("region_matches", float),
        ("player_skill", lambda: 1.0),
        ("player_maps", float),
    ]:
        upgraded[key] = defaultdict(factory, dict(upgraded.get(key, {})))
    map_stats = defaultdict(
        lambda: {
            "wins": 0.0,
            "maps": 0.0,
            "rating_sum": 0.0,
            "picks": 0.0,
            "deciders": 0.0,
            "last_date": None,
            "patch": "",
        }
    )
    map_stats.update(upgraded.get("map_stats", {}))
    upgraded["map_stats"] = map_stats
    return upgraded


def normalize_lineup_identity(value, player_id=None) -> str:
    numeric_id = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
    if pd.notna(numeric_id):
        return f"id:{int(numeric_id)}"
    text = str(value or "")
    if text.startswith("id:") and text[3:].isdigit():
        return text
    return normalize_player_name(text)


def _as_timestamp(value) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp if pd.notna(timestamp) else None


def _elapsed_days(previous, current) -> float:
    previous_ts = _as_timestamp(previous)
    current_ts = _as_timestamp(current)
    if previous_ts is None or current_ts is None:
        return 0.0
    return max(0.0, float((current_ts - previous_ts).total_seconds() / 86400.0))


def _half_life_weight(days: float, half_life_days: float) -> float:
    return 0.5 ** (max(0.0, days) / half_life_days)


def _lineups_by_match(cleaned: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    lineups = {}
    if cleaned.empty:
        return lineups
    for (match_key, team), group in cleaned.groupby(["match_key", "team"], sort=False):
        identity_columns = ["player"]
        if "player_id" in group.columns:
            identity_columns.append("player_id")
        lineups[(match_key, team)] = {
            normalize_lineup_identity(row.get("player"), row.get("player_id"))
            for _, row in group.drop_duplicates(subset=identity_columns).iterrows()
        }
    return lineups


def _player_observations_by_match(cleaned: pd.DataFrame) -> dict[str, list[dict]]:
    observations = defaultdict(list)
    if cleaned.empty or "rating_for_model" not in cleaned.columns:
        return observations
    identity_columns = ["match_key", "team", "player"]
    for (match_key, team, player), group in cleaned.groupby(identity_columns, sort=False):
        player_id = group["player_id"].dropna().iloc[0] if "player_id" in group and group["player_id"].notna().any() else None
        observations[match_key].append(
            {
                "identity": normalize_lineup_identity(player, player_id),
                "team": team,
                "rating": float(group["rating_for_model"].mean()),
                "maps": float(group["map_id"].nunique()) if "map_id" in group else float(len(group)),
            }
        )
    return observations


def _map_team_rows(cleaned: pd.DataFrame) -> pd.DataFrame:
    if cleaned.empty or "map_id" not in cleaned.columns:
        return pd.DataFrame()
    scoped = cleaned[cleaned["map_id"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame()

    rows = []
    for (match_key, map_id, team), group in scoped.groupby(
        ["match_key", "map_id", "team"],
        sort=False,
    ):
        team_score = group["map_team_score"].dropna().iloc[0] if "map_team_score" in group and group["map_team_score"].notna().any() else None
        opp_score = group["map_opp_score"].dropna().iloc[0] if "map_opp_score" in group and group["map_opp_score"].notna().any() else None
        map_win = float(team_score > opp_score) if team_score is not None and opp_score is not None else float(group["is_winner"].astype(float).max())
        rows.append(
            {
                "match_key": match_key,
                "map_id": map_id,
                "map_name": group["map_name"].dropna().iloc[0] if "map_name" in group and group["map_name"].notna().any() else "Unknown",
                "team": team,
                "opponent": group["opponent"].dropna().iloc[0] if "opponent" in group and group["opponent"].notna().any() else "",
                "map_pick_team": (
                    group["map_pick_team"].dropna().iloc[0]
                    if "map_pick_team" in group and group["map_pick_team"].notna().any()
                    else ""
                ),
                "map_pick_type": (
                    str(group["map_pick_type"].dropna().iloc[0]).lower()
                    if "map_pick_type" in group and group["map_pick_type"].notna().any()
                    else "unknown"
                ),
                "map_win": map_win,
                "team_score": float(team_score) if team_score is not None else None,
                "opp_score": float(opp_score) if opp_score is not None else None,
                "team_avg_rating": float(group["rating_for_model"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _continuity(current: set[str], previous: set[str] | None) -> float:
    if not current or not previous:
        return 0.5
    return min(1.0, len(current & previous) / max(5, len(current)))


def _map_stat(state: dict, team: str, map_name: str) -> dict:
    return state["map_stats"].get(
        (team, map_name),
        {
            "wins": 0.0,
            "maps": 0.0,
            "rating_sum": 0.0,
            "picks": 0.0,
            "deciders": 0.0,
            "last_date": None,
            "patch": "",
        },
    )


def _decayed_map_stat(
    stat: dict,
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    decay = _half_life_weight(
        _elapsed_days(stat.get("last_date"), as_of_date),
        MAP_POOL_HALF_LIFE_DAYS,
    )
    known_patch = str(stat.get("patch") or "").strip()
    current_patch = str(current_patch or "").strip()
    patch_factor = PATCH_CARRY if known_patch and current_patch and known_patch != current_patch else 1.0
    factor = decay * patch_factor
    return {
        **stat,
        "wins": float(stat.get("wins", 0.0)) * factor,
        "maps": float(stat.get("maps", 0.0)) * factor,
        "rating_sum": float(stat.get("rating_sum", 0.0)) * factor,
        "picks": float(stat.get("picks", 0.0)) * factor,
        "deciders": float(stat.get("deciders", 0.0)) * factor,
    }


def _map_rate(stat: dict) -> float:
    return (stat["wins"] + MAP_PRIOR_GAMES * 0.5) / (stat["maps"] + MAP_PRIOR_GAMES)


def _map_rating(stat: dict) -> float:
    return (stat["rating_sum"] + MAP_PRIOR_GAMES * MAP_PRIOR_RATING) / (
        stat["maps"] + MAP_PRIOR_GAMES
    )


def _effective_team_rating(state: dict, team: str) -> float:
    team_rating = float(state["elo"].get(team, INITIAL_ELO))
    region = str(state["team_regions"].get(team, "") or "")
    region_rating = float(state["region_elo"].get(region, INITIAL_ELO)) if region else INITIAL_ELO
    return team_rating + REGION_ELO_WEIGHT * (region_rating - INITIAL_ELO)


def _map_elo_at(
    state: dict,
    team: str,
    map_name: str,
    as_of_date=None,
) -> float:
    team_rating = _effective_team_rating(state, team)
    key = (team, map_name)
    stored_rating = float(state["map_elo"].get(key, team_rating))
    decay = _half_life_weight(
        _elapsed_days(state["map_elo_last_date"].get(key), as_of_date),
        MAP_POOL_HALF_LIFE_DAYS,
    )
    return team_rating + (stored_rating - team_rating) * decay


def _player_skill_at(state: dict, identity: str, as_of_date=None) -> tuple[float, float]:
    maps = float(state["player_maps"].get(identity, 0.0))
    skill = float(state["player_skill"].get(identity, 1.0))
    decay = _half_life_weight(
        _elapsed_days(state["player_last_date"].get(identity), as_of_date),
        PLAYER_SKILL_HALF_LIFE_DAYS,
    )
    skill = 1.0 + (skill - 1.0) * decay
    reliability = maps / (maps + 8.0) if maps > 0 else 0.0
    return skill, reliability


def _lineup_skill_context(
    state: dict,
    lineup_a: set[str],
    lineup_b: set[str],
    as_of_date=None,
) -> dict:
    def summary(lineup: set[str]) -> tuple[float, float]:
        if not lineup:
            return 1.0, 0.0
        values = [_player_skill_at(state, identity, as_of_date) for identity in lineup]
        return (
            float(sum(value for value, _ in values) / len(values)),
            float(sum(reliability for _, reliability in values) / len(values)),
        )

    rating_a, reliability_a = summary(lineup_a)
    rating_b, reliability_b = summary(lineup_b)
    return {
        "lineup_rating_diff": rating_a - rating_b,
        "lineup_reliability": min(reliability_a, reliability_b),
        "team_lineup_rating": rating_a,
        "opponent_lineup_rating": rating_b,
    }


def _prepare_team_state(
    state: dict,
    team: str,
    as_of_date,
    continuity: float,
    current_patch: str = "",
) -> None:
    rating = float(state["elo"].get(team, INITIAL_ELO))
    previous_date = state["elo_last_date"].get(team)
    days = _elapsed_days(previous_date, as_of_date)
    rating = INITIAL_ELO + (rating - INITIAL_ELO) * _half_life_weight(
        days,
        TEAM_ELO_HALF_LIFE_DAYS,
    )
    state["elo_freshness"][team] = (
        _half_life_weight(days, TEAM_ELO_HALF_LIFE_DAYS)
        if previous_date is not None
        else 0.0
    )

    as_of = _as_timestamp(as_of_date)
    previous_season = state["elo_season"].get(team)
    current_season = as_of.year if as_of is not None else previous_season
    if previous_season is not None and current_season != previous_season:
        rating = INITIAL_ELO + (rating - INITIAL_ELO) * SEASON_CARRY

    roster_carry = MIN_ROSTER_CARRY + (1.0 - MIN_ROSTER_CARRY) * max(
        0.0,
        min(1.0, continuity),
    )
    rating = INITIAL_ELO + (rating - INITIAL_ELO) * roster_carry

    previous_patch = str(state["last_patch"].get(team, "") or "").strip()
    current_patch = str(current_patch or "").strip()
    if previous_patch and current_patch and previous_patch != current_patch:
        rating = INITIAL_ELO + (rating - INITIAL_ELO) * PATCH_CARRY

    sos_days = _elapsed_days(state["sos_last_date"].get(team), as_of_date)
    sos_decay = _half_life_weight(sos_days, TEAM_ELO_HALF_LIFE_DAYS)
    state["opponent_elo_sum"][team] *= sos_decay
    state["opponent_count"][team] *= sos_decay
    state["elo"][team] = rating
    if as_of is not None:
        state["elo_last_date"][team] = as_of
        state["sos_last_date"][team] = as_of
        state["elo_season"][team] = current_season
    if current_patch:
        state["last_patch"][team] = current_patch


def _map_pool_context(
    state: dict,
    team_a: str,
    team_b: str,
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    map_names = sorted(
        {
            map_name
            for team, map_name in state["map_stats"]
            if team in {team_a, team_b}
        }
    )
    if not map_names:
        return {
            "map_pool_win_rate_diff": 0.0,
            "map_pool_rating_diff": 0.0,
            "map_pool_elo_diff": 0.0,
            "map_pool_maps": 0.0,
            "map_pool_reliability": 0.0,
            "map_probabilities": {},
            "map_signal_probabilities": {},
            "map_reliabilities": {},
            "map_team_games": {},
            "map_opponent_games": {},
            "likely_map_order": [],
            "map_selection_reliability": 0.0,
        }

    elo_a = _effective_team_rating(state, team_a)
    elo_b = _effective_team_rating(state, team_b)
    base_logit = _logit(elo_probability(elo_a, elo_b))
    weighted_win_diff = 0.0
    weighted_rating_diff = 0.0
    weighted_elo_diff = 0.0
    total_weight = 0.0
    map_probabilities = {}
    map_signal_probabilities = {}
    map_reliabilities = {}
    map_team_games = {}
    map_opponent_games = {}
    map_evidence = {}
    map_selection_scores = {}
    pick_evidence_total = 0.0
    total_a = 0
    total_b = 0

    for map_name in map_names:
        stat_a = _decayed_map_stat(
            _map_stat(state, team_a, map_name),
            as_of_date=as_of_date,
            current_patch=current_patch,
        )
        stat_b = _decayed_map_stat(
            _map_stat(state, team_b, map_name),
            as_of_date=as_of_date,
            current_patch=current_patch,
        )
        count_a = stat_a["maps"]
        count_b = stat_b["maps"]
        total_a += count_a
        total_b += count_b
        evidence = map_history_reliability(count_a, count_b)
        weight = max(0.15, evidence)
        win_diff = _map_rate(stat_a) - _map_rate(stat_b)
        rating_diff = _map_rating(stat_a) - _map_rating(stat_b)
        map_elo_a = _map_elo_at(state, team_a, map_name, as_of_date)
        map_elo_b = _map_elo_at(state, team_b, map_name, as_of_date)
        map_elo_diff = (map_elo_a - map_elo_b) / ELO_SCALE
        weighted_win_diff += weight * win_diff
        weighted_rating_diff += weight * rating_diff
        weighted_elo_diff += weight * map_elo_diff
        total_weight += weight
        map_elo_logit = _logit(elo_probability(map_elo_a, map_elo_b))
        map_signal_probability = _sigmoid(
            map_elo_logit
            + 1.35 * win_diff
            + 0.90 * rating_diff
        )
        map_probabilities[map_name] = hierarchical_map_probability(
            _sigmoid(base_logit),
            map_signal_probability,
            evidence,
        )
        map_signal_probabilities[map_name] = map_signal_probability
        map_reliabilities[map_name] = evidence
        map_team_games[map_name] = count_a
        map_opponent_games[map_name] = count_b
        map_evidence[map_name] = count_a + count_b
        pick_evidence = float(stat_a.get("picks", 0.0)) + float(stat_b.get("picks", 0.0))
        decider_evidence = float(stat_a.get("deciders", 0.0)) + float(stat_b.get("deciders", 0.0))
        joint_appearance = math.sqrt((count_a + 0.25) * (count_b + 0.25))
        map_selection_scores[map_name] = (
            joint_appearance + 1.4 * pick_evidence + 0.45 * decider_evidence
        )
        pick_evidence_total += pick_evidence

    min_maps = min(total_a, total_b)
    likely_map_order = sorted(
        map_probabilities,
        key=lambda name: (
            map_selection_scores[name],
            map_evidence[name],
        ),
        reverse=True,
    )
    selection_reliability = (
        min(1.0, pick_evidence_total / 12.0)
        if pick_evidence_total > 0
        else min(0.45, min_maps / 45.0)
    )
    return {
        "map_pool_win_rate_diff": weighted_win_diff / total_weight,
        "map_pool_rating_diff": weighted_rating_diff / total_weight,
        "map_pool_elo_diff": weighted_elo_diff / total_weight,
        "map_pool_maps": float(min_maps),
        "map_pool_reliability": min(1.0, min_maps / 20.0),
        "map_probabilities": map_probabilities,
        "map_signal_probabilities": map_signal_probabilities,
        "map_reliabilities": map_reliabilities,
        "map_team_games": map_team_games,
        "map_opponent_games": map_opponent_games,
        "likely_map_order": likely_map_order,
        "map_selection_reliability": selection_reliability,
    }


def _context_features(
    state: dict,
    team_a: str,
    team_b: str,
    lineup_a: set[str],
    lineup_b: set[str],
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    team_elo_a = float(state["elo"].get(team_a, INITIAL_ELO))
    team_elo_b = float(state["elo"].get(team_b, INITIAL_ELO))
    elo_a = _effective_team_rating(state, team_a)
    elo_b = _effective_team_rating(state, team_b)
    region_a = str(state["team_regions"].get(team_a, "") or "")
    region_b = str(state["team_regions"].get(team_b, "") or "")
    region_elo_a = float(state["region_elo"].get(region_a, INITIAL_ELO)) if region_a else INITIAL_ELO
    region_elo_b = float(state["region_elo"].get(region_b, INITIAL_ELO)) if region_b else INITIAL_ELO
    region_matches_a = float(state["region_matches"].get(region_a, 0.0)) if region_a else 0.0
    region_matches_b = float(state["region_matches"].get(region_b, 0.0)) if region_b else 0.0
    cross_region = bool(region_a and region_b and region_a != region_b)
    count_a = state["opponent_count"].get(team_a, 0)
    count_b = state["opponent_count"].get(team_b, 0)
    sos_a = state["opponent_elo_sum"].get(team_a, 0.0) / count_a if count_a else INITIAL_ELO
    sos_b = state["opponent_elo_sum"].get(team_b, 0.0) / count_b if count_b else INITIAL_ELO
    continuity_a = _continuity(lineup_a, state["lineups"].get(team_a))
    continuity_b = _continuity(lineup_b, state["lineups"].get(team_b))
    map_context = _map_pool_context(
        state,
        team_a,
        team_b,
        as_of_date=as_of_date,
        current_patch=current_patch,
    )
    elo_freshness = min(
        float(state["elo_freshness"].get(team_a, 0.0)),
        float(state["elo_freshness"].get(team_b, 0.0)),
    )
    patch_a = str(state["last_patch"].get(team_a, "") or "").strip()
    patch_b = str(state["last_patch"].get(team_b, "") or "").strip()
    same_patch_context = float(not current_patch or not patch_a or not patch_b or (patch_a == patch_b == current_patch))
    lineup_context = _lineup_skill_context(
        state,
        lineup_a,
        lineup_b,
        as_of_date=as_of_date,
    )

    return {
        "elo_diff": (elo_a - elo_b) / ELO_SCALE,
        "team_elo_diff": (team_elo_a - team_elo_b) / ELO_SCALE,
        "elo_probability": elo_probability(elo_a, elo_b),
        "elo_matches": float(min(count_a, count_b)),
        "elo_reliability": min(1.0, min(count_a, count_b) / 20.0),
        "minimum_elo_freshness": elo_freshness,
        "strength_of_schedule_diff": (sos_a - sos_b) / ELO_SCALE,
        "region_elo_diff": (region_elo_a - region_elo_b) / ELO_SCALE,
        "region_reliability": min(1.0, min(region_matches_a, region_matches_b) / 12.0) if cross_region else 1.0,
        "cross_region_match": float(cross_region),
        "roster_continuity_diff": continuity_a - continuity_b,
        "minimum_roster_continuity": min(continuity_a, continuity_b),
        "same_patch_context": same_patch_context,
        "team_roster_continuity": continuity_a,
        "opponent_roster_continuity": continuity_b,
        "team_region": region_a,
        "opponent_region": region_b,
        **lineup_context,
        **map_context,
    }


def build_sequential_team_context(
    cleaned: pd.DataFrame,
    team_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    state = _new_state()
    if team_matches.empty:
        return pd.DataFrame(), state

    lineups = _lineups_by_match(cleaned)
    player_observations = _player_observations_by_match(cleaned)
    map_rows = _map_team_rows(cleaned)
    map_groups = {
        match_key: group
        for match_key, group in map_rows.groupby("match_key", sort=False)
    } if not map_rows.empty else {}
    rows = []
    ordered = team_matches.sort_values(["match_date_sort", "match_id"], na_position="first")

    for match_key, group in ordered.groupby("match_key", sort=False):
        if len(group) != 2:
            continue
        first, second = group.iloc[0], group.iloc[1]
        team_a = first["team"]
        team_b = second["team"]
        region_a = str(first.get("team_region", "") or "").strip()
        region_b = str(second.get("team_region", "") or "").strip()
        if region_a:
            state["team_regions"][team_a] = region_a
        if region_b:
            state["team_regions"][team_b] = region_b
        lineup_a = lineups.get((match_key, team_a), set())
        lineup_b = lineups.get((match_key, team_b), set())
        match_date = first.get("match_date_sort")
        current_patch = str(first.get("patch", "") or "").strip()
        continuity_a = _continuity(lineup_a, state["lineups"].get(team_a))
        continuity_b = _continuity(lineup_b, state["lineups"].get(team_b))
        _prepare_team_state(
            state,
            team_a,
            match_date,
            continuity_a,
            current_patch=current_patch,
        )
        _prepare_team_state(
            state,
            team_b,
            match_date,
            continuity_b,
            current_patch=current_patch,
        )
        context_a = _context_features(
            state,
            team_a,
            team_b,
            lineup_a,
            lineup_b,
            as_of_date=match_date,
            current_patch=current_patch,
        )
        context_b = _context_features(
            state,
            team_b,
            team_a,
            lineup_b,
            lineup_a,
            as_of_date=match_date,
            current_patch=current_patch,
        )
        context_a_row = dict(context_a)
        context_b_row = dict(context_b)
        rows.extend(
            [
                {"match_key": match_key, "team": team_a, "opponent": team_b, **context_a_row},
                {"match_key": match_key, "team": team_b, "opponent": team_a, **context_b_row},
            ]
        )

        elo_a = float(state["elo"][team_a])
        elo_b = float(state["elo"][team_b])
        effective_elo_a = _effective_team_rating(state, team_a)
        effective_elo_b = _effective_team_rating(state, team_b)
        expected_a = elo_probability(effective_elo_a, effective_elo_b)
        outcome_a = float(first["team_win"])
        margin_multiplier = 1.0 + 0.12 * min(3.0, abs(float(first.get("score_margin", 0.0))))
        importance = float(first.get("match_importance", 1.0) or 1.0)
        importance = max(0.75, min(1.15, importance))
        change = ELO_K * importance * margin_multiplier * (outcome_a - expected_a)
        state["elo"][team_a] = elo_a + change
        state["elo"][team_b] = elo_b - change
        state["opponent_elo_sum"][team_a] += elo_b
        state["opponent_elo_sum"][team_b] += elo_a
        state["opponent_count"][team_a] += 1
        state["opponent_count"][team_b] += 1

        if region_a and region_b and region_a != region_b:
            region_rating_a = float(state["region_elo"].get(region_a, INITIAL_ELO))
            region_rating_b = float(state["region_elo"].get(region_b, INITIAL_ELO))
            region_expected_a = elo_probability(region_rating_a, region_rating_b)
            region_change = REGION_ELO_K * importance * (outcome_a - region_expected_a)
            state["region_elo"][region_a] = region_rating_a + region_change
            state["region_elo"][region_b] = region_rating_b - region_change
            state["region_matches"][region_a] += 1.0
            state["region_matches"][region_b] += 1.0

        match_map_rows = map_groups.get(match_key, pd.DataFrame())
        if not match_map_rows.empty:
            for _, paired_maps in match_map_rows.groupby("map_id", sort=False):
                if len(paired_maps) != 2:
                    continue
                map_a_rows = paired_maps[paired_maps["team"] == team_a]
                map_b_rows = paired_maps[paired_maps["team"] == team_b]
                if map_a_rows.empty or map_b_rows.empty:
                    continue
                map_a = map_a_rows.iloc[0]
                map_b = map_b_rows.iloc[0]
                map_name = map_a["map_name"]
                map_key_a = (team_a, map_name)
                map_key_b = (team_b, map_name)
                map_elo_a = _map_elo_at(state, team_a, map_name, match_date)
                map_elo_b = _map_elo_at(state, team_b, map_name, match_date)
                map_expected_a = elo_probability(map_elo_a, map_elo_b)
                map_margin = abs(float(map_a.get("team_score", 0.0) or 0.0) - float(map_a.get("opp_score", 0.0) or 0.0))
                map_multiplier = 1.0 + 0.025 * min(8.0, map_margin)
                map_change = MAP_ELO_K * map_multiplier * (float(map_a["map_win"]) - map_expected_a)
                state["map_elo"][map_key_a] = map_elo_a + map_change
                state["map_elo"][map_key_b] = map_elo_b - map_change
                state["map_elo_matches"][map_key_a] += 1.0
                state["map_elo_matches"][map_key_b] += 1.0
                state["map_elo_last_date"][map_key_a] = _as_timestamp(match_date)
                state["map_elo_last_date"][map_key_b] = _as_timestamp(match_date)

        for _, map_row in match_map_rows.iterrows():
            stat = state["map_stats"][(map_row["team"], map_row["map_name"])]
            decayed = _decayed_map_stat(
                stat,
                as_of_date=match_date,
                current_patch=current_patch,
            )
            stat.update(decayed)
            stat["wins"] += float(map_row["map_win"])
            stat["maps"] += 1.0
            stat["rating_sum"] += float(map_row["team_avg_rating"])
            if str(map_row.get("map_pick_team", "")) == str(map_row["team"]):
                stat["picks"] += 1.0
            if str(map_row.get("map_pick_type", "")).lower() == "decider":
                stat["deciders"] += 1.0
            stat["last_date"] = _as_timestamp(match_date)
            if current_patch:
                stat["patch"] = current_patch
        for observation in player_observations.get(match_key, []):
            identity = observation["identity"]
            prior_skill, _ = _player_skill_at(state, identity, match_date)
            maps_played = max(1.0, float(observation.get("maps", 1.0)))
            update_weight = 1.0 - (1.0 - 0.18) ** maps_played
            state["player_skill"][identity] = prior_skill + update_weight * (
                float(observation["rating"]) - prior_skill
            )
            state["player_maps"][identity] += maps_played
            state["player_last_date"][identity] = _as_timestamp(match_date)
        if lineup_a:
            state["lineups"][team_a] = lineup_a
        if lineup_b:
            state["lineups"][team_b] = lineup_b

    return pd.DataFrame(rows), state


def current_team_context(
    cleaned: pd.DataFrame,
    team_matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    lineup_a: set[str] | None = None,
    lineup_b: set[str] | None = None,
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    _, state = build_sequential_team_context(cleaned, team_matches)
    return team_context_from_state(
        state,
        team_a,
        team_b,
        lineup_a=lineup_a,
        lineup_b=lineup_b,
        as_of_date=as_of_date,
        current_patch=current_patch,
    )


def prepare_team_state_for_prediction(
    state: dict,
    lineups_by_team: dict[str, set[str]] | None = None,
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    """Prepare every team once for batch, read-only matchup evaluation."""
    prepared = _upgrade_state(deepcopy(state))
    lineups_by_team = lineups_by_team or {}
    if as_of_date is None:
        dates = [
            _as_timestamp(value)
            for value in prepared["elo_last_date"].values()
            if _as_timestamp(value) is not None
        ]
        as_of_date = max(dates) if dates else pd.Timestamp.now(tz="UTC")

    teams = set(prepared["elo"]) | set(lineups_by_team)
    for team in teams:
        lineup = lineups_by_team.get(team, prepared["lineups"].get(team, set()))
        continuity = _continuity(lineup, prepared["lineups"].get(team))
        _prepare_team_state(
            prepared,
            team,
            as_of_date,
            continuity,
            current_patch=current_patch,
        )
    return prepared


def team_context_from_prepared_state(
    state: dict,
    team_a: str,
    team_b: str,
    lineup_a: set[str] | None = None,
    lineup_b: set[str] | None = None,
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    """Read matchup features from a state prepared by prepare_team_state_for_prediction."""
    state = _upgrade_state(state)
    if lineup_a is None:
        lineup_a = state["lineups"].get(team_a, set())
    if lineup_b is None:
        lineup_b = state["lineups"].get(team_b, set())
    return _context_features(
        state,
        team_a,
        team_b,
        lineup_a,
        lineup_b,
        as_of_date=as_of_date,
        current_patch=current_patch,
    )


def freeze_team_state(state: dict) -> dict:
    return {
        "elo": dict(state["elo"]),
        "opponent_elo_sum": dict(state["opponent_elo_sum"]),
        "opponent_count": dict(state["opponent_count"]),
        "map_stats": {
            key: dict(value)
            for key, value in state["map_stats"].items()
        },
        "map_elo": dict(state["map_elo"]),
        "map_elo_matches": dict(state["map_elo_matches"]),
        "map_elo_last_date": dict(state["map_elo_last_date"]),
        "region_elo": dict(state["region_elo"]),
        "region_matches": dict(state["region_matches"]),
        "team_regions": dict(state["team_regions"]),
        "player_skill": dict(state["player_skill"]),
        "player_maps": dict(state["player_maps"]),
        "player_last_date": dict(state["player_last_date"]),
        "lineups": {
            team: set(lineup)
            for team, lineup in state["lineups"].items()
        },
        "elo_last_date": dict(state["elo_last_date"]),
        "elo_freshness": dict(state["elo_freshness"]),
        "elo_season": dict(state["elo_season"]),
        "last_patch": dict(state["last_patch"]),
        "sos_last_date": dict(state["sos_last_date"]),
    }


def team_context_from_state(
    state: dict,
    team_a: str,
    team_b: str,
    lineup_a: set[str] | None = None,
    lineup_b: set[str] | None = None,
    as_of_date=None,
    current_patch: str = "",
) -> dict:
    state = _upgrade_state(state)
    if lineup_a is None:
        lineup_a = state["lineups"].get(team_a, set())
    if lineup_b is None:
        lineup_b = state["lineups"].get(team_b, set())
    if as_of_date is None:
        dates = [
            _as_timestamp(value)
            for value in state["elo_last_date"].values()
            if _as_timestamp(value) is not None
        ]
        as_of_date = max(dates) if dates else pd.Timestamp.now(tz="UTC")
    continuity_a = _continuity(lineup_a, state["lineups"].get(team_a))
    continuity_b = _continuity(lineup_b, state["lineups"].get(team_b))
    _prepare_team_state(
        state,
        team_a,
        as_of_date,
        continuity_a,
        current_patch=current_patch,
    )
    _prepare_team_state(
        state,
        team_b,
        as_of_date,
        continuity_b,
        current_patch=current_patch,
    )
    return _context_features(
        state,
        team_a,
        team_b,
        lineup_a,
        lineup_b,
        as_of_date=as_of_date,
        current_patch=current_patch,
    )
