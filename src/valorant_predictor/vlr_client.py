import argparse
import re
import time
from datetime import datetime
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import BASE_URL, MATCHES_CSV, NEWS_CSV, REQUEST_HEADERS, ROSTERS_CSV, TEAM_PAGES
from .features.form_calculations import filter_matches_by_time


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    return session


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def absolute_url(href: str) -> str:
    return urljoin(BASE_URL, href)


def match_id_from_url(url: str) -> int | None:
    match = re.search(r"vlr\.gg/(\d+)|/(\d+)/", str(url))
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


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


def get_team_match_urls(session: requests.Session, team_url: str, limit: int) -> list[str]:
    soup = get_soup(session, team_url)
    urls = []
    for link in soup.select("a.wf-card.fc-flex.m-item, a.wf-card"):
        href = link.get("href", "")
        if not href.startswith("/") or "vs" not in href:
            continue
        full_url = absolute_url(href)
        if full_url not in urls:
            urls.append(full_url)
        if len(urls) >= limit:
            break
    return urls


def team_profile_url_from_matches_url(team_url: str) -> str:
    return team_url.replace("/team/matches/", "/team/")


def _player_id_from_href(href: str) -> int | None:
    match = re.search(r"/player/(\d+)/", href or "")
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
    for team_name, team_url in pages.items():
        print(f"Collecting roster for {team_name}")
        try:
            records.extend(parse_team_roster_page(session, team_name, team_url))
        except requests.RequestException as exc:
            print(f"Skipping roster for {team_name}: {exc}")
        time.sleep(pause_seconds)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
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
    url: str,
    winner: str | None,
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
        else:
            parts = list(cols[0].stripped_strings)
            player = parts[0]

        if row_team not in teams:
            return None

        opponent = teams[1] if row_team == teams[0] else teams[0]

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
            "opponent": opponent,
            "winner": winner,
            "player": player,
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
    team_names = [node.get_text(strip=True) for node in soup.select(".match-header .wf-title-med")]
    if len(team_names) != 2:
        return []

    teams = (team_names[0], team_names[1])
    match_id = match_id_from_url(url)
    match_date = match_date_from_page(soup)
    score1, score2 = _score_from_header(soup)
    winner = None
    if score1 is not None and score2 is not None:
        winner = teams[0] if score1 > score2 else teams[1]

    records = []
    tables = soup.select(".vm-stats-game table.wf-table-inset")
    for table_index, table in enumerate(tables):
        row_team = teams[table_index % 2]
        for row in table.select("tbody tr"):
            parsed = _parse_player_row(row, teams, url, winner, row_team=row_team)
            if parsed:
                parsed["match_id"] = match_id
                parsed["match_date"] = match_date
                parsed["team_score"] = score1 if parsed["team"] == teams[0] else score2
                parsed["opp_score"] = score2 if parsed["team"] == teams[0] else score1
                records.append(parsed)
    return records


def scrape_matches(
    output_csv: str = MATCHES_CSV,
    limit_per_team: int = 25,
    team_pages: dict[str, str] | None = None,
    pause_seconds: float = 1.2,
    season_year: int | None = None,
    recent_days: int | None = None,
) -> pd.DataFrame:
    session = make_session()
    pages = team_pages or TEAM_PAGES
    match_urls = []

    for team_name, team_url in pages.items():
        print(f"Collecting matches for {team_name}")
        for url in get_team_match_urls(session, team_url, limit_per_team):
            if url not in match_urls:
                match_urls.append(url)
        time.sleep(pause_seconds)

    records = []
    for index, url in enumerate(match_urls, 1):
        print(f"[{index}/{len(match_urls)}] {url}")
        try:
            records.extend(parse_match_page(session, url))
        except requests.RequestException as exc:
            print(f"Skipping {url}: {exc}")
        time.sleep(pause_seconds)

    df = pd.DataFrame(records)
    if not df.empty and (season_year is not None or recent_days is not None):
        df = filter_matches_by_time(df, season_year=season_year, recent_days=recent_days)
    df.to_csv(output_csv, index=False)
    return df


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


def scrape_news(output_csv: str = NEWS_CSV, pages: int = 3, pause_seconds: float = 1.0) -> pd.DataFrame:
    session = make_session()
    records = []
    for page in range(1, pages + 1):
        print(f"Collecting VLR news page {page}")
        records.extend(parse_news_index_page(session, page))
        time.sleep(pause_seconds)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape VLR match stats and news.")
    parser.add_argument("--matches", action="store_true", help="Scrape match/player stats.")
    parser.add_argument("--news", action="store_true", help="Scrape VLR news.")
    parser.add_argument("--rosters", action="store_true", help="Scrape current VLR team rosters.")
    parser.add_argument("--limit-per-team", type=int, default=25)
    parser.add_argument("--news-pages", type=int, default=3)
    parser.add_argument("--season-year", type=int, help="Keep only matches from this year when dates are available.")
    parser.add_argument("--recent-days", type=int, help="Keep only matches from the last N days when dates are available.")
    args = parser.parse_args()

    if not args.matches and not args.news and not args.rosters:
        parser.error("Choose --matches, --news, --rosters, or a combination.")

    if args.matches:
        df = scrape_matches(
            limit_per_team=args.limit_per_team,
            season_year=args.season_year,
            recent_days=args.recent_days,
        )
        print(f"Saved {len(df)} match stat rows to {MATCHES_CSV}")
    if args.news:
        df = scrape_news(pages=args.news_pages)
        print(f"Saved {len(df)} news rows to {NEWS_CSV}")
    if args.rosters:
        df = scrape_rosters()
        print(f"Saved {len(df)} roster rows to {ROSTERS_CSV}")


if __name__ == "__main__":
    main()
