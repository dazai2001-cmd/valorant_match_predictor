import re
import unicodedata
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd

from .config import BASE_URL, TEAM_PAGES, TEAMS_CSV
from .vlr_client import absolute_url, get_soup, make_session


REGISTRY_COLUMNS = [
    "team",
    "team_id",
    "matches_url",
    "profile_url",
    "source",
    "league",
    "region",
    "tier",
    "season_year",
    "active",
    "ranking_page",
    "added_at",
    "resolved_at",
]

VCT_TIER1_SEASON = 2026

VCT_TIER1_TEAMS = {
    2026: {
        "VCT Americas": [
            "Leviatan",
            "G2 Esports",
            "NRG",
            "FURIA",
            "100 Thieves",
            "KRU Esports",
            "MIBR",
            "LOUD",
            "ENVY",
            "Sentinels",
            "Cloud9",
            "Evil Geniuses",
        ],
        "VCT EMEA": [
            "FUT Esports",
            "Team Vitality",
            "Team Heretics",
            "BBL Esports",
            "Eternal Fire",
            "Fnatic",
            "Team Liquid",
            "Gentle Mates",
            "GIANTX",
            "Natus Vincere",
            "Karmine Corp",
            "PCFIC Esports",
        ],
        "VCT Pacific": [
            "Paper Rex",
            "Nongshim RedForce",
            "T1",
            "FULL SENSE",
            "Global Esports",
            "Rex Regum Qeon",
            "DRX",
            "DetonatioN FocusMe",
            "Gen.G",
            "Team Secret",
            "ZETA DIVISION",
            "VARREL",
        ],
        "VCT China": [
            "Edward Gaming",
            "Xi Lai Gaming",
            "All Gamers",
            "Dragon Ranger Gaming",
            "Bilibili Gaming",
            "JD Gaming",
            "TYLOO",
            "FunPlus Phoenix",
            "Titan Esports Club",
            "Trace Esports",
            "Nova Esports",
            "Wolves Esports",
        ],
    }
}

def slugify_team(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "team"


def team_key(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_name.lower())


def team_matches_url(team_id: int | str, name: str, slug: str | None = None) -> str:
    team_slug = slug or slugify_team(name)
    return f"{BASE_URL}/team/matches/{team_id}/{team_slug}/"


def team_profile_url(team_id: int | str, name: str, slug: str | None = None) -> str:
    team_slug = slug or slugify_team(name)
    return f"{BASE_URL}/team/{team_id}/{team_slug}/"


def iter_vct_tier1_teams(season_year: int = VCT_TIER1_SEASON) -> list[dict]:
    teams_by_league = VCT_TIER1_TEAMS.get(season_year)
    if not teams_by_league:
        raise ValueError(f"No VCT Tier 1 seed list is configured for {season_year}.")

    rows = []
    for league, teams in teams_by_league.items():
        for team in teams:
            rows.append(
                {
                    "team": team,
                    "league": league,
                    "region": league,
                    "tier": "tier1",
                    "season_year": str(season_year),
                    "active": "true",
                }
            )
    return rows


def league_for_seed_team(team: str, season_year: int = VCT_TIER1_SEASON) -> str:
    lookup = {team_key(row["team"]): row["league"] for row in iter_vct_tier1_teams(season_year)}
    return lookup.get(team_key(team), "")


def clean_team_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def clean_bool(value, default: bool = True) -> str:
    if pd.isna(value) or value == "":
        return "true" if default else "false"
    text = str(value).strip().lower()
    return "true" if text in {"1", "true", "yes", "y", "on"} else "false"


def registry_dedupe_key(row: pd.Series) -> str:
    key = team_key(str(row.get("team", "")))
    if key:
        return f"name:{key}"
    team_id = clean_team_id(row.get("team_id"))
    return f"id:{team_id}" if team_id else ""


def normalize_registry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    output = df.copy()
    for column in REGISTRY_COLUMNS:
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].fillna("")
    output["team_id"] = output["team_id"].map(clean_team_id)
    output["active"] = output["active"].map(clean_bool)
    output["tier"] = output["tier"].replace("", "tier1")
    output["season_year"] = output["season_year"].replace("", str(VCT_TIER1_SEASON))
    output = output[output["team"].notna() & (output["team"].astype(str).str.strip() != "")].copy()
    output["_dedupe_key"] = output.apply(registry_dedupe_key, axis=1)
    output["_has_url"] = output["matches_url"].astype(str).str.strip() != ""
    output = output.sort_values(["_dedupe_key", "_has_url", "added_at"])
    output = output.drop_duplicates(subset=["_dedupe_key"], keep="last")
    output = output.drop(columns=["_dedupe_key", "_has_url"])
    output = output[REGISTRY_COLUMNS]
    return output.sort_values("team").reset_index(drop=True)


def load_team_registry(path: str = TEAMS_CSV) -> pd.DataFrame:
    rows = [
        {
            "team": name,
            "team_id": re.search(r"/team/matches/(\d+)/", url).group(1),
            "matches_url": url,
            "profile_url": url.replace("/team/matches/", "/team/"),
            "source": "config",
            "league": league_for_seed_team(name),
            "region": league_for_seed_team(name),
            "tier": "tier1",
            "season_year": str(VCT_TIER1_SEASON),
            "active": "true",
            "ranking_page": "",
            "added_at": "",
            "resolved_at": "",
        }
        for name, url in TEAM_PAGES.items()
        if re.search(r"/team/matches/(\d+)/", url)
    ]
    try:
        saved = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        saved = pd.DataFrame()

    combined = pd.concat([pd.DataFrame(rows), saved], ignore_index=True)
    return normalize_registry(combined)


def save_team_registry(df: pd.DataFrame, path: str = TEAMS_CSV) -> None:
    normalize_registry(df).to_csv(path, index=False)


def active_tier1_registry(path: str = TEAMS_CSV, season_year: int = VCT_TIER1_SEASON) -> pd.DataFrame:
    registry = load_team_registry(path)
    if registry.empty:
        return registry
    return registry[
        (registry["tier"].astype(str).str.lower() == "tier1")
        & (registry["active"].astype(str).str.lower() == "true")
        & (registry["season_year"].astype(str) == str(season_year))
    ].reset_index(drop=True)


def active_tier1_team_names(path: str = TEAMS_CSV, season_year: int = VCT_TIER1_SEASON) -> list[str]:
    registry = active_tier1_registry(path, season_year)
    if registry.empty:
        return [row["team"] for row in iter_vct_tier1_teams(season_year)]
    return sorted(registry["team"].dropna().astype(str).unique())


def registry_team_pages(path: str = TEAMS_CSV) -> dict[str, str]:
    registry = active_tier1_registry(path)
    if registry.empty:
        return dict(TEAM_PAGES)
    registry = registry[
        registry["team"].notna()
        & registry["matches_url"].notna()
        & (registry["matches_url"].astype(str).str.strip() != "")
    ]
    return dict(zip(registry["team"], registry["matches_url"]))


def search_vlr_teams(query: str, limit: int = 8) -> list[dict]:
    session = make_session()
    soup = get_soup(session, f"{BASE_URL}/search/?q={quote_plus(query)}&type=teams")
    results = []
    for link in soup.select("a.search-item[href*='/search/r/team/']"):
        href = link.get("href", "")
        match = re.search(r"/search/r/team/(\d+)/", href)
        title = link.select_one(".search-item-title")
        if not match or not title:
            continue
        name = title.get_text(" ", strip=True)
        name = re.sub(r"\s+\(inactive.*?\)", "", name).strip()
        is_inactive = "inactive" in title.get_text(" ", strip=True).lower()
        team_id = match.group(1)
        results.append(
            {
                "team": name,
                "team_id": team_id,
                "matches_url": team_matches_url(team_id, name),
                "profile_url": team_profile_url(team_id, name),
                "is_inactive": is_inactive,
                "search_url": absolute_url(href),
            }
        )
        if len(results) >= limit:
            break
    return results


def best_vlr_result(team: str, results: list[dict]) -> dict | None:
    if not results:
        return None
    target = team_key(team)
    active_results = [row for row in results if not row.get("is_inactive")]
    for row in active_results:
        if team_key(row.get("team", "")) == target:
            return row
    for row in active_results:
        result_key = team_key(row.get("team", ""))
        if target in result_key or result_key in target:
            return row
    return active_results[0] if active_results else results[0]


def seed_vct_tier1_teams(
    path: str = TEAMS_CSV,
    season_year: int = VCT_TIER1_SEASON,
    resolve_missing: bool = True,
) -> tuple[pd.DataFrame, dict]:
    current = load_team_registry(path)
    existing_by_team = {team_key(row["team"]): row for _, row in current.iterrows()} if not current.empty else {}

    rows = []
    unresolved = []
    resolved_count = 0
    resolution_error = ""
    now = datetime.utcnow().isoformat(timespec="seconds")

    for seed in iter_vct_tier1_teams(season_year):
        existing = existing_by_team.get(team_key(seed["team"]))
        team = seed["team"]
        team_id = clean_team_id(existing.get("team_id")) if existing is not None else ""
        matches_url = str(existing.get("matches_url", "")).strip() if existing is not None else ""
        profile_url = str(existing.get("profile_url", "")).strip() if existing is not None else ""
        resolved_at = str(existing.get("resolved_at", "")).strip() if existing is not None else ""

        if resolve_missing and (not team_id or not matches_url) and not resolution_error:
            try:
                result = best_vlr_result(team, search_vlr_teams(team, limit=5))
            except Exception as exc:
                resolution_error = str(exc)
                result = None
            if result:
                team_id = result["team_id"]
                matches_url = result["matches_url"]
                profile_url = result["profile_url"]
                resolved_at = now
                resolved_count += 1

        if not team_id or not matches_url:
            unresolved.append(team)

        rows.append(
            {
                "team": team,
                "team_id": team_id,
                "matches_url": matches_url,
                "profile_url": profile_url,
                "source": "vct_tier1_seed",
                "league": seed["league"],
                "region": seed["region"],
                "tier": "tier1",
                "season_year": str(season_year),
                "active": "true",
                "ranking_page": "",
                "added_at": str(existing.get("added_at", "")).strip() if existing is not None else now,
                "resolved_at": resolved_at,
            }
        )

    registry = normalize_registry(pd.DataFrame(rows))
    save_team_registry(registry, path)
    meta = {
        "season_year": season_year,
        "seeded_count": len(rows),
        "resolved_count": int((registry["matches_url"].astype(str).str.strip() != "").sum()),
        "newly_resolved_count": resolved_count,
        "unresolved": unresolved,
        "resolution_error": resolution_error,
    }
    return registry, meta


def discover_ranked_teams(path: str = TEAMS_CSV) -> pd.DataFrame:
    registry, _ = seed_vct_tier1_teams(path=path, resolve_missing=True)
    return registry
